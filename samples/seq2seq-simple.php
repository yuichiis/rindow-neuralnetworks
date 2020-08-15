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

        $this->embedding = $builder->Embedding($vocab_size, $word_vect_size);
        $this->lstm = $builder->SimpleRNN($recurrent_units);
    }

    public function build(array $inputShape=null, array $options=null) : array
    {
        $inputShape=$this->normalizeInputShapetShape($inputShape);
        $inputShape = $this->registerLayer($this->embedding,$inputShape);
        $inputShape = $this->registerLayer($this->lstm,$inputShape);
        $this->outputShape = $inputShape;
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
        [$dWordvect,$dStates]=$this->lstm->backward($dOutputs,$dStates);
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

        $this->embedding = $builder->Embedding($vocab_size, $word_vect_size);
        $this->lstm = $builder->SimpleRNN($recurrent_units);
        $this->dense = $builder->Dense($dense_units);
    }

    public function build(array $inputShape=null, array $options=null) : array
    {
        $inputShape=$this->normalizeInputShapetShape($inputShape);
        $inputShape = $this->registerLayer($this->embedding,$inputShape);
        $inputShape = $this->registerLayer($this->lstm,$inputShape);
        $inputShape = $this->registerLayer($this->dense,$inputShape);
        $this->outputShape = $inputShape;
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
    }
    
    protected function buildLayers(array $options=null) : void
    {
        $this->registerLayer($this->encoder);
        $this->registerLayer($this->decoder);
    }

    protected function forwardStep(NDArray $inputs, NDArray $trues=null, bool $training=null) : NDArray
    {
        [$dummy,$states] = $this->encoder->forward($inputs,null,$training);
        [$outputs,$dummy] = $this->decoder->forward($trues,$states,$training);
        $this->outputShape = $outputs->shape();
        return $outputs;
    }
    
    protected function backwardStep(NDArray $dout) : NDArray
    {
        [$dummy,$dStates] = $this->decoder->backward($dout,null);
        [$dInputs,$dStates] = $this->encoder->backward($K->zeros($this->outputShape),$dStates);
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
        var_dump($numbers->shape());
        for($i=0;$i<$corp_size;$i++){
            $num = $numbers[$i];
            var_dump($num);
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
        var_dump($str);
        $sseq = str_split($str);
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
    
    public function loadData($path=null)
    {
        $corp_size = 10000;
        $length = 5;
        if($path==null){
            $path='dec2hex-dataset.pkl';
        }
        if(file_exists($path)){
            $pkl = file_get_contents($path);
            $dataset = unserialize($pkl);
        }else{
            $dataset = $this->generate($corp_size,$length);
            $pkl = serialize($dataset);
            file_put_contents($path,$dataset);
        }
        return $dataset;
    }

}

$mo = new MatrixOperator();
$backend = new Backend($mo);
$nn = new NeuralNetworks($mo,$backend);
$dataset = new DecHexDataset($mo);
[$dec,$hex]=$dataset->loadData();
$train_inputs = $dec[[0,9899]];
$train_target = $hex[[0,9899]];
$test_input = $dec[[9900,9999]];
$test_target = $hex[[9900,9999]];
$input_length = $train_inputs->shape()[1];
[$iv,$tv,$input_dic,$target_dic]=$dataset->dicts();
$input_vocab_size = count($input_dic);
$target_vocab_size = count($target_dic);

$seq2seq2 = new Seq2seq($backend,$nn,[
    'input_length'=>$input_length,
    'input_vocab_size'=>$input_vocab_size,
    'target_vocab_size'=>$input_vocab_size,
]);