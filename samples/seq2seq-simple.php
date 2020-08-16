<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\NeuralNetworks\Support\GenericUtils;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Layer\AbstractRNNLayer;
use Rindow\NeuralNetworks\Model\AbstractModel;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;


class Encoder extends AbstractRNNLayer
{
    protected $backend;
    protected $builder;
    protected $vocabSize;
    protected $wordVectSize;
    protected $recurrentUnits;
    protected $embedding;
    protected $lstm;

    public function __construct(
        $backend,
        $builder,
        $input_length,
        $vocab_size,
        $word_vect_size,
        $recurrent_units
    )
    {
        $this->backend = $backend;
        $this->inputShape=[$input_length];
        $this->vocabSize = $vocab_size;
        $this->wordVectSize = $word_vect_size;
        $this->recurrentSize = $recurrent_units;

        $this->embedding = $builder->layers()->Embedding($vocab_size, $word_vect_size);
        $this->lstm = $builder->layers()->SimpleRNN($recurrent_units,[
                'return_state'=>true
                ]);
    }

    public function build(array $inputShape=null, array $options=null) : array
    {
        $inputShape=$this->normalizeInputShape($inputShape);
        $inputShape = $this->registerLayer($this->embedding,$inputShape);
        $inputShape = $this->registerLayer($this->lstm,$inputShape);
        $this->outputShape = $inputShape;
        $this->statesShapes = $this->lstm->statesShapes();
        return $this->outputShape;
    }

    public function getConfig() : array
    {
        return [
            'builder'=>true,
            'vocab_size'=>$this->vocabSize,
            'word_vec_size'=>$this->wordVecSize,
            'recurrent_units'=>$this->recurrentUnits,
            ];
    }

    protected function call(NDArray $inputs,bool $training, array $initalStates=null, array $options=null)
    {
        $wordvect = $this->embedding->forward($inputs,$training);
        [$outputs,$states]=$this->lstm->forward($wordvect,$training,$initalStates);
        return [$outputs,$states];
    }
    
    protected function differentiate(NDArray $dOutputs, array $dNextStates=null)
    {
        [$dWordvect,$dStates]=$this->lstm->backward($dOutputs,$dNextStates);
        $dInputs = $this->embedding->backward($dWordvect);
        return [$dInputs,$dStates];
    }
}

class Decoder extends AbstractRNNLayer
{
    protected $backend;
    protected $builder;
    protected $vocabSize;
    protected $wordVectSize;
    protected $recurrentUnits;
    protected $denseUnits;
    protected $embedding;
    protected $lstm;
    protected $dense;
    
    public function __construct(
        $backend,
        $builder,
        $input_length,
        $vocab_size,
        $word_vect_size,
        $recurrent_units,
        $dense_units
    )
    {
        $this->backend = $backend;
        $this->inputShape=[$input_length];
        $this->vocabSize = $vocab_size;
        $this->wordVectSize = $word_vect_size;
        $this->recurrentSize = $recurrent_units;
        $this->denseUnits = $dense_units;

        $this->embedding = $builder->layers()->Embedding($vocab_size, $word_vect_size);
        $this->lstm = $builder->layers()->SimpleRNN(
            $recurrent_units,[
                'return_state'=>true,
                'return_sequence'=>true,
                ]);
        $this->dense = $builder->layers()->Dense($dense_units);
    }

    public function build(array $inputShape=null, array $options=null) : array
    {
        $inputShape=$this->normalizeInputShape($inputShape);
        $inputShape = $this->registerLayer($this->embedding,$inputShape);
        $inputShape = $this->registerLayer($this->lstm,$inputShape);
        $inputShape = $this->registerLayer($this->dense,$inputShape);
        $this->outputShape = $inputShape;
        $this->statesShapes = $this->lstm->statesShapes();
        
        return $this->outputShape;
    }

    public function getConfig() : array
    {
        return [
            'builder'=>true,
            'vocab_size'=>$this->vocabSize,
            'word_vec_size'=>$this->wordVecSize,
            'recurrent_units'=>$this->recurrentUnits,
            'dense_units'=>$this->denseUnits,
            ];
    }

    protected function call(NDArray $inputs,bool $training, array $initalStates=null, array $options=null)
    {
        $wordvect = $this->embedding->forward($inputs,$training);
        [$outputs,$states]=$this->lstm->forward($wordvect,$training,$initalStates);
        $outputs=$this->dense->forward($outputs,$training);
        return [$outputs,$states];
    }
    
    protected function differentiate(NDArray $dOutputs, array $dNextStates=null)
    {
        $dOutputs = $this->dense->backward($dOutputs);
        [$dWordvect,$dStates]=$this->lstm->backward($dOutputs);
        $dInputs = $this->embedding->backward($dWordvect);
        return [$dInputs,$dStates];
    }
}

class Seq2seq extends AbstractModel
{
    use GenericUtils;
    protected $encode;
    protected $decode;
    protected $encoutShape;
    
    public function __construct($backend,$builder,array $options=null)
    {
        extract($this->extractArgs([
            'input_length'=>null,
            'input_vocab_size'=>null,
            'target_vocab_size'=>null,
            'word_vect_size'=>8,
            'recurrent_units'=>256,
            'dense_units'=>256,
        ],$options));
        parent::__construct($backend,$builder,$builder->utils()->HDA());
        $this->encoder = new Encoder(
            $backend,$builder,
            $input_length,
            $input_vocab_size,
            $word_vect_size,
            $recurrent_units
        );
        $this->decoder = new Decoder(
            $backend,$builder,
            $input_length,
            $target_vocab_size,
            $word_vect_size,
            $recurrent_units,
            $dense_units
        );
        $this->out = $builder->layers()->Activation('softmax');
        $this->setLastLayer($this->out);
    }
    
    protected function buildLayers(array $options=null) : void
    {
        $this->registerLayer($this->encoder);
        $shape = $this->registerLayer($this->decoder);
        $this->registerLayer($this->out,$shape);
    }

    protected function forwardStep(NDArray $inputs, NDArray $trues=null, bool $training=null) : NDArray
    {
        [$dummy,$states] = $this->encoder->forward($inputs,$training,null);
        $this->encoutShape = $dummy->shape();
        
        [$outputs,$dummy] = $this->decoder->forward($trues,$training,$states);
        $outputs = $this->out->forward($outputs,$training);
        return $outputs;
    }
    
    protected function backwardStep(NDArray $dout) : NDArray
    {
        $K = $this->backend;
        $dout = $this->out->backward($dout);
        [$dummy,$dStates] = $this->decoder->backward($dout,null);
        [$dInputs,$dStates] = $this->encoder->backward($K->zeros($this->encoutShape),$dStates);
        return $dInputs;
    }
    
    public function translate(NDArray $sentence)
    {
        $K = $this->backend;
        $this->setShapeInspection(false);
        [$dmy,$states]=$this->encoder->forward($sentence,null,$training=false);
        $vocId = 0;
        $targetSentence =[];
        for($i=0;$i<$inputLength;$i++){
            $in = $K->array([[$vocId]]);
            [$predictions,$dmy] = $this->decoder->forward($in,$states,$training=false);
            $vocId = $K->argMax($predictions);
            $targetSentence[]=$vocId;
        }
        $this->setShapeInspection(true);
        return $K->array($targetSentence);
    }
}

class DecHexDataset
{
    public function __construct($mo)
    {
        $this->mo = $mo;
        $this->vocab_input = ['0','1','2','3','4','5','6','7','8','9',' ','@'];
        $this->vocab_target = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F',' ','@'];
        $this->dict_input = array_flip($this->vocab_input);
        $this->dict_target = array_flip($this->vocab_target);
    }
    
    public function dicts()
    {
        return [
            $this->vocab_input,
            $this->vocab_target,
            $this->dict_input,
            $this->dict_target,
        ];
    }
    
    public function generate($corp_size,$length)
    {
        $sequence = $this->mo->zeros([$corp_size,$length]);
        $target = $this->mo->zeros([$corp_size,$length]);
        $numbers = $this->mo->random()->choice($corp_size,$corp_size);
        for($i=0;$i<$corp_size;$i++){
            $num = $numbers[$i];
            $dec = strval($num);
            $hex = dechex($num);
            $this->str2seq(
                $dec,
                $this->dict_input,
                $sequence[$i]);
            $this->str2seq(
                '@'.$hex,
                $this->dict_target,
                $target[$i]);
        }
        return [$sequence,$target];
    }
    
    public function str2seq(
        string $str,
        array $dic,
        NDArray $buf)
    {
        $sseq = str_split(strtoupper($str));
        $len = count($sseq);
        $sp = $dic[' '];
        $bufsz=$buf->size();
        for($i=0;$i<$bufsz;$i++){
            if($i<$len)
                $buf[$i]=$dic[$sseq[$i]];
            else
                $buf[$i]=$sp;
        }
    }
    
    public function seq2str(
        NDArray $buf,
        array $dic
        )
    {
        $str = '';
        $bufsz=$buf->size();
        for($i=0;$i<$bufsz;$i++){
            $str .= $dic[$buf[$i]];
        }
        return $str;
    }
    
    public function translate($model,$str)
    {
        $inputs = $this->mo->zeros([1,$this->length]);
        $this->str2seq(
            $str,$this->dict_input,$buf[0]);
        $target = $model->translate($inputs);
        return $this->seq2str(
            $target[0],$this->vocab_target);
            );
    }

    public function loadData($corp_size,$path=null)
    {
        $this->length = strlen(strval($corp_size));
        if($path==null){
            $path='dec2hex-dataset.pkl';
        }
        if(file_exists($path)){
            $pkl = file_get_contents($path);
            $dataset = unserialize($pkl);
        }else{
            $dataset = $this->generate($corp_size,$this->length);
            $pkl = serialize($dataset);
            file_put_contents($path,$pkl);
        }
        return $dataset;
    }

}

$corp_size = 40000;
$test_size = 100;
$mo = new MatrixOperator();
$backend = new Backend($mo);
$nn = new NeuralNetworks($mo,$backend);
$dataset = new DecHexDataset($mo);
[$dec,$hex]=$dataset->loadData($corp_size);
$train_inputs = $dec[[0,$corp_size-$test_size-1]];
$train_target = $hex[[0,$corp_size-$test_size-1]];
$test_input = $dec[[$corp_size-$test_size,$corp_size-1]];
$test_target = $hex[[$corp_size-$test_size,$corp_size-1]];
$input_length = $train_inputs->shape()[1];
[$iv,$tv,$input_dic,$target_dic]=$dataset->dicts();
$input_vocab_size = count($input_dic);
$target_vocab_size = count($target_dic);

$seq2seq = new Seq2seq($backend,$nn,[
    'input_length'=>$input_length,
    'input_vocab_size'=>$input_vocab_size,
    'target_vocab_size'=>$target_vocab_size,
]);

$seq2seq->compile([
    'optimizer'=>$nn->optimizers()->Adam(),
    ]);
$history = $seq2seq->fit($train_inputs,$train_target,
    ['epochs'=>1,'batch_size'=>64,'validation_data'=>[$test_input,$test_target]]);

$samples = ['10','255','1024'];
foreach ($samples as $value) {
    $target = $dataset->translate(
        $seq2seq,$value);
    echo "[$value]=>[$target]\n";
}