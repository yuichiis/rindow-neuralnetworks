<?php
require __DIR__.'/../vendor/autoload.php';

//use InvalidArgumentException;
use Rindow\NeuralNetworks\Support\GenericUtils;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Layer\AbstractRNNLayer;
use Rindow\NeuralNetworks\Model\AbstractModel;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

$TRAINING_SIZE = 5000;
$DIGITS = 2;
$REVERSE = True;

class Encoder extends AbstractRNNLayer
{
    protected $backend;
    protected $builder;
    protected $vocabSize;
    protected $wordVectSize;
    protected $recurrentUnits;
    protected $embedding;
    protected $rnn;

    public function __construct(
        $backend,
        $builder,
        string $rnn,
        int $input_length,
        int $vocab_size,
        int $word_vect_size,
        int $recurrent_units
    )
    {
        $this->backend = $backend;
        $this->inputShape=[$input_length];
        $this->vocabSize = $vocab_size;
        $this->wordVectSize = $word_vect_size;
        $this->recurrentSize = $recurrent_units;

        $this->embedding = $builder->layers()->Embedding($vocab_size, $word_vect_size);
        $this->rnnName = $rnn;
        if($rnn=='simple') {
            $this->rnn = $builder->layers()->SimpleRNN(
                $recurrent_units,[
                    'return_state'=>true,
                ]);
        } elseif($rnn=='lstm') {
            $this->rnn = $builder->layers()->LSTM(
                $recurrent_units,[
                    'return_state'=>true,
                ]);
        } elseif($rnn=='gru') {
            $this->rnn = $builder->layers()->GRU(
                $recurrent_units,[
                    'return_state'=>true,
                ]);
        } else {
            throw new InvalidArgumentException('unknown rnn type: '.$rnn);
        }
    }

    public function build(array $inputShape=null, array $options=null) : array
    {
        $inputShape=$this->normalizeInputShape($inputShape);
        $inputShape = $this->registerLayer($this->embedding,$inputShape);
        $inputShape = $this->registerLayer($this->rnn,$inputShape);
        $this->outputShape = $inputShape;
        $this->statesShapes = $this->rnn->statesShapes();
        return $this->outputShape;
    }

    public function getConfig() : array
    {
        return [
            'builder'=>true,
            'rnn'=>$this->rnnName,
            'vocab_size'=>$this->vocabSize,
            'word_vec_size'=>$this->wordVecSize,
            'recurrent_units'=>$this->recurrentUnits,
            ];
    }

    protected function call(NDArray $inputs,bool $training, array $initalStates=null, array $options=null)
    {
        $wordvect = $this->embedding->forward($inputs,$training);
        [$outputs,$states]=$this->rnn->forward($wordvect,$training,$initalStates);
        return [$outputs,$states];
    }

    protected function differentiate(NDArray $dOutputs, array $dNextStates=null)
    {
        [$dWordvect,$dStates]=$this->rnn->backward($dOutputs,$dNextStates);
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
    protected $rnn;
    protected $dense;

    public function __construct(
        $backend,
        $builder,
        string $rnn,
        int $input_length,
        int $vocab_size,
        int $word_vect_size,
        int $recurrent_units,
        int $dense_units
    )
    {
        $this->backend = $backend;
        $this->inputShape=[$input_length];
        $this->vocabSize = $vocab_size;
        $this->wordVectSize = $word_vect_size;
        $this->recurrentSize = $recurrent_units;
        $this->denseUnits = $dense_units;

        $this->embedding = $builder->layers()->Embedding($vocab_size, $word_vect_size);
        $this->rnnName = $rnn;
        if($rnn=='simple') {
            $this->rnn = $builder->layers()->SimpleRNN(
                $recurrent_units,[
                    'return_state'=>true,
                    'return_sequences'=>true,
                ]);
        } elseif($rnn=='lstm') {
            $this->rnn = $builder->layers()->LSTM(
                $recurrent_units,[
                    'return_state'=>true,
                    'return_sequences'=>true,
                ]);
        } elseif($rnn=='gru') {
            $this->rnn = $builder->layers()->GRU(
                $recurrent_units,[
                    'return_state'=>true,
                    'return_sequences'=>true,
                ]);
        } else {
            throw new InvalidArgumentException('unknown rnn type: '.$rnn);
        }
        $this->dense = $builder->layers()->Dense($dense_units);
    }

    public function build(array $inputShape=null, array $options=null) : array
    {
        $inputShape=$this->normalizeInputShape($inputShape);
        $inputShape = $this->registerLayer($this->embedding,$inputShape);
        $inputShape = $this->registerLayer($this->rnn,$inputShape);
        $inputShape = $this->registerLayer($this->dense,$inputShape);
        $this->outputShape = $inputShape;
        $this->statesShapes = $this->rnn->statesShapes();

        return $this->outputShape;
    }

    public function getConfig() : array
    {
        return [
            'builder'=>true,
            'rnn'=>$this->rnnName,
            'vocab_size'=>$this->vocabSize,
            'word_vec_size'=>$this->wordVecSize,
            'recurrent_units'=>$this->recurrentUnits,
            'dense_units'=>$this->denseUnits,
        ];
    }

    protected function call(NDArray $inputs,bool $training, array $initalStates=null, array $options=null)
    {
        $wordvect = $this->embedding->forward($inputs,$training);
        [$outputs,$states]=$this->rnn->forward($wordvect,$training,$initalStates);
        $outputs=$this->dense->forward($outputs,$training);
        return [$outputs,$states];
    }

    protected function differentiate(NDArray $dOutputs, array $dNextStates=null)
    {
        $dOutputs = $this->dense->backward($dOutputs);
        [$dWordvect,$dStates]=$this->rnn->backward($dOutputs);
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
            'rnn'=>null,
            'input_length'=>null,
            'input_vocab_size'=>null,
            'output_length'=>null,
            'target_vocab_size'=>null,
            'word_vect_size'=>8,
            'recurrent_units'=>256,
            'dense_units'=>256,
            'start_voc_id'=>0,
        ],$options));
        parent::__construct($backend,$builder,$builder->utils()->HDA());
        $this->encoder = new Encoder(
            $backend,$builder,
            $rnn,
            $input_length,
            $input_vocab_size,
            $word_vect_size,
            $recurrent_units
        );
        $this->decoder = new Decoder(
            $backend,$builder,
            $rnn,
            $output_length,
            $target_vocab_size,
            $word_vect_size,
            $recurrent_units,
            $dense_units
        );
        $this->out = $builder->layers()->Activation('softmax');
        $this->setLastLayer($this->out);
        $this->startVocId = $start_voc_id;
        $this->outputLength = $output_length;
    }

    protected function buildLayers(array $options=null) : void
    {
        $this->registerLayer($this->encoder);
        $shape = $this->registerLayer($this->decoder);
        $this->registerLayer($this->out,$shape);
    }

    protected function shiftSentence(
        NDArray $sentence)
    {
        $K = $this->backend;
        $result = $K->zerosLike($sentence);
        [$batches,$length] = $sentence->shape();
        for($batch=0;$batch<$batches;$batch++){
            $source = $sentence[$batch][[0,$length-2]];
            $dest = $result[$batch][[1,$length-1]];
            $result[$batch][0]=$this->startVocId;
            $K->copy($source,$dest);
        }
        return $result;
    }

    protected function forwardStep(NDArray $inputs, NDArray $trues=null, bool $training=null) : NDArray
    {
        $K = $this->backend;
        if($trues===null) {
            [$outputs,$states] = $this->encoder->forward($inputs,$training,null);
            return $outputs;
        }
        [$dummy,$states] = $this->encoder->forward($inputs,$training,null);
        $this->encoutShape = $dummy->shape();
        $dec_inputs = $this->shiftSentence($trues);
        [$outputs,$dummy] = $this->decoder->forward($dec_inputs,$training,$states);
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
        $inputLength = $sentence->size();
        $sentence = $sentence->reshape([1,$inputLength]);
        $this->setShapeInspection(false);
        [$dmy,$states]=$this->encoder->forward($sentence,$training=false);
        $vocId = $this->startVocId;
        $targetSentence =[];
        for($i=0;$i<$this->outputLength;$i++){
            $in = $K->array([[$vocId]]);
            [$predictions,$states] = $this->decoder->forward($in,$training=false,$states);
            $vocId = $K->argMax($predictions);
            $targetSentence[]=$vocId;
        }
        $this->setShapeInspection(true);
        return $K->array($targetSentence);
    }
}

class NumAdditionDataset
{
    public function __construct($mo,int $corpus_max,int $digits)
    {
        $this->mo = $mo;
        $this->corpus_max = $corpus_max;
        $this->digits = $digits;
        #$this->reverse = $reverse;
        $this->vocab_input  = ['0','1','2','3','4','5','6','7','8','9','+',' '];
        $this->vocab_target = ['0','1','2','3','4','5','6','7','8','9','+',' '];
        $this->dict_input  = array_flip($this->vocab_input);
        $this->dict_target = array_flip($this->vocab_target);
        $this->input_length = $digits*2+1;
        $this->output_length = $digits+1;
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

    public function generate()
    {
        $max_num = pow(10,$this->digits);
        $max_sample = $max_num ** 2;
        $numbers = $this->mo->random()->choice(
            $max_sample,$max_sample,$replace=false);
        $questions = [];
        $dups = [];
        $size = 0;
        for($i=0;$i<$max_sample;$i++) {
            $num = $numbers[$i];
            $x1 = (int)floor($num / $max_num);
            $x2 = (int)($num % $max_num);
            if($x1>$x2) {
                [$x1,$x2] = [$x2,$x1];
            }
            $question = $x1.'+'.$x2;
            if(array_key_exists($question,$questions)) {
                #echo $question.',';
                $dups[$question] += 1;
                continue;
            }
            $dups[$question] = 1;
            $questions[$question] = strval($x1+$x2);
            $size++;
            if($size >= $this->corpus_max)
                break;
        }
        unset($numbers);
        $sequence = $this->mo->zeros([$size,$this->input_length],NDArray::int32);
        $target = $this->mo->zeros([$size,$this->output_length],NDArray::int32);
        $i = 0;
        foreach($questions as $question=>$answer) {
            $question = str_pad($question, $this->input_length);
            $answer = str_pad($answer, $this->output_length);
            $this->str2seq(
                $question,
                $this->dict_input,
                $sequence[$i]);
            $this->str2seq(
                $answer,
                $this->dict_target,
                $target[$i]);
            $i++;
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

    //public function translate($model,$str)
    //{
    //    $inputs = $this->mo->zeros([1,$this->length]);
    //    $this->str2seq(
    //        $str,$this->dict_input,$inputs[0]);
    //    $target = $model->translate($inputs);
    //    return $this->seq2str(
    //        $target,$this->vocab_target
    //        );
    //}

    public function loadData($path=null)
    {
        if($path==null){
            $path='numaddition-dataset.pkl';
        }
        if(file_exists($path)){
            $pkl = file_get_contents($path);
            $dataset = unserialize($pkl);
        }else{
            $dataset = $this->generate();
            $pkl = serialize($dataset);
            file_put_contents($path,$pkl);
        }
        return $dataset;
    }

}

$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);
$backend = $nn->backend();
$plt = new Plot(null,$mo);

//$rnn = 'simple';
//$rnn = 'lstm';
$rnn = 'gru';
$input_length  = $DIGITS*2 + 1;
$output_length = $DIGITS + 1;

$dataset = new NumAdditionDataset($mo,$TRAINING_SIZE,$DIGITS);
echo "Generating data...\n";
[$questions,$answers] = $dataset->loadData();
$corpus_size = $questions->shape()[0];
echo "Total questions: ". $corpus_size."\n";
[$input_voc,$target_voc,$input_dic,$target_dic]=$dataset->dicts();


# Explicitly set apart 10% for validation data that we never train over.
$split_at = $corpus_size - (int)floor($corpus_size / 10);
$x_train = $questions[[0,$split_at-1]];
$x_val   = $questions[[$split_at,$corpus_size-1]];
$y_train = $answers[[0,$split_at-1]];
$y_val   = $answers[[$split_at,$corpus_size-1]];

echo "train,test: ".$x_train->shape()[0].",".$y_train->shape()[0]."\n";

$seq2seq = new Seq2seq($backend,$nn,[
    'rnn'=>$rnn,
    'input_length'=>$input_length,
    'input_vocab_size'=>count($input_dic),
    'output_length'=>$output_length,
    'target_vocab_size'=>count($target_dic),
    'start_voc_id'=>$dataset->dict_target[' '],
    'word_vect_size'=>16,
    'recurrent_units'=>128,
    'dense_units'=>128,
]);

echo "Compile model...\n";
$seq2seq->compile([
    'loss'=>'sparse_categorical_crossentropy',
    'optimizer'=>'adam',
]);
$seq2seq->summary();

$epochs = 30;
$batch_size = 32;

echo "Train model...\n";
$history = $seq2seq->fit(
    $x_train,$y_train,
    ['epochs'=>$epochs,'batch_size'=>$batch_size,'validation_data'=>[$x_val,$y_val]]);

for($i=0;$i<10;$i++) {
    $idx = $mo->random()->randomInt($corpus_size);
    $question = $questions[$idx];

    #$input = $question->reshape([1,$input_length]);
    #$predict = $seq2seq->predict($input);
    #$predict_seq = $mo->argMax($predict[0]->reshape([$output_length,count($target_dic)]),$axis=1);
    $predict_seq = $seq2seq->translate($question);

    $predict_str = $dataset->seq2str($predict_seq,$target_voc);
    $question_str = $dataset->seq2str($question,$input_voc);
    $answer_str = $dataset->seq2str($answers[$idx],$target_voc);
    $correct = ($predict_str==$answer_str) ? '*' : ' ';
    echo "$question_str=$predict_str : $correct $answer_str\n";
}

$plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
$plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
$plt->plot($mo->array($history['loss']),null,null,'loss');
$plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
$plt->legend();
$plt->title('seq2seq-basic');
$plt->show();
