<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\NeuralNetworks\Support\GenericUtils;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Layer\AbstractRNNLayer;
use Rindow\NeuralNetworks\Model\AbstractModel;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Data\Sequence\Tokenizer;
use Rindow\NeuralNetworks\Data\Sequence\Preprocessor;

# Download the file
class EngFraDataset
{
    protected $baseUrl = 'http://www.manythings.org/anki/';
    protected $downloadFile = 'fra-eng.zip';

    public function __construct($mo,$inputTokenizer=null,$targetTokenizer=null)
    {
        $this->mo = $mo;
        $this->datasetDir = $this->getDatasetDir();
        if(!file_exists($this->datasetDir)) {
            @mkdir($this->datasetDir,0777,true);
        }
        $this->saveFile = $this->datasetDir . "/fra-eng.pkl";
        $this->preprocessor = new Preprocessor($mo);
    }

    protected function getDatasetDir()
    {
        return sys_get_temp_dir().'/rindow/nn/datasets/fra-eng';
    }

    protected function download($filename)
    {
        $filePath = $this->datasetDir . "/" . $filename;

        if(!file_exists($filePath)){
            $this->console("Downloading " . $filename . " ... ");
            copy($this->baseUrl.$filename, $filePath);
            $this->console("Done\n");
        }

        $memberfile = 'fra.txt';
        $path = $this->datasetDir.'/'.$memberfile;
        if(file_exists($path)){
            return $path;
        }
        $this->console("Extract to:".$this->datasetDir.'/..'."\n");
        $files = [$memberfile];
        $zip = new ZipArchive();
        $zip->open($filePath);
        $zip->extractTo($this->datasetDir);
        $zip->close();
        $this->console("Done\n");

        return $path;
    }

    # Converts the unicode file to ascii
    #def unicode_to_ascii(self,s):
    #    return ''.join(c for c in unicodedata.normalize('NFD', s)
    #    if unicodedata.category(c) != 'Mn')

    public function preprocessSentence($w)
    {
        $w = '<start> '.$w.' <end>';
        return $w;
    }

    public function createDataset($path, $numExamples)
    {
        $contents = file_get_contents($path);
        if($contents==false) {
            throw new InvalidArgumentException('file not found: '.$path);
        }
        $lines = explode("\n",trim($contents));
        unset($contents);
        $trim = function($w) { return trim($w); };
        $enSentences = [];
        $spSentences = [];
        foreach ($lines as $line) {
            if($numExamples!==null) {
                $numExamples--;
                if($numExamples<0)
                    break;
            }
            $blocks = explode("\t",$line);
            $blocks = array_map($trim,$blocks);
            $en = $this->preprocessSentence($blocks[0]);
            $sp = $this->preprocessSentence($blocks[1]);
            $enSentences[] = $en;
            $spSentences[] = $sp;
        }
        return [$enSentences,$spSentences];
    }

    public function tokenize($lang,$numWords=null,$tokenizer=null)
    {
        if($tokenizer==null) {
            $tokenizer = new Tokenizer($this->mo,[
                'num_words'=>$numWords,
                'filters'=>"\"#$%&()*+,-./:;=@[\\]^_`{|}~\t\n",
                'specials'=>"?.!,¿",
            ]);
        }
        $tokenizer->fitOnTexts($lang);
        $sequences = $tokenizer->textsToSequences($lang);
        $tensor = $this->preprocessor->padSequences($sequences,['padding'=>'post']);
        return [$tensor, $tokenizer];
    }

    protected function console($message)
    {
        fwrite(STDERR,$message);
    }

    public function loadData(
        string $path=null, int $numExamples=null, int $numWords=null)
    {
        if($path==null) {
            $path = $this->download($this->downloadFile);
        }
        # creating cleaned input, output pairs
        [$targ_lang, $inp_lang] = $this->createDataset($path, $numExamples);

        [$input_tensor, $inp_lang_tokenizer] = $this->tokenize($inp_lang,$numWords);
        [$target_tensor, $targ_lang_tokenizer] = $this->tokenize($targ_lang,$numWords);
        $numInput = $input_tensor->shape()[0];
        $choice = $this->mo->random()->choice($numInput,$numInput,$replace=false);
        $input_tensor = $this->shuffle($input_tensor,$choice);
        $target_tensor = $this->shuffle($target_tensor,$choice);

        return [$input_tensor, $target_tensor, $inp_lang_tokenizer, $targ_lang_tokenizer];
    }

    public function shuffle(NDArray $tensor, NDArray $choice) : NDArray
    {
        $result = $this->mo->zerosLike($tensor);
        $size = $tensor->shape()[0];
        for($i=0;$i<$size;$i++) {
            $this->mo->la()->copy($tensor[$choice[$i]],$result[$i]);
        }
        return $result;
    }

    public function convert($lang, NDArray $tensor) : void
    {
        $size = $tensor->shape()[0];
        for($i=0;$t<$size;$t++) {
            $t = $tensor[$i];
            if($t!=0)
                echo sprintf("%d ----> %s\n", $t, $lang->index_word[$t]);
        }
    }
}

class Encoder extends AbstractRNNLayer
{
    public function __construct(
        $backend,
        $builder,
        int $vocabSize,
        int $wordVectSize,
        int $units
        )
    {
        $this->vocabSize = $vocabSize;
        $this->wordVectSize = $wordVectSize;
        $this->units = $units;
        $this->embedding = $builder->layers()->Embedding($vocabSize,$wordVectSize);
        $this->rnn = $builder->layers()->GRU(
            $units,
            ['return_state'=>true,'return_sequences'=>true]
        );
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
            'vocab_size'=>$this->vocabSize,
            'word_vec_size'=>$this->wordVecSize,
            'units'=>$this->units,
            ];
    }

    protected function call(
        NDArray $inputs,
        bool $training,
        array $initial_state=null,
        array $options=null
        ) : NDArray
    {
        $wordVect = $this->embedding->forward($inputs);
        [$outputs,$states] = $this->rnn->forward(
            $wordVect,$training,$initial_state);
        return [$outputs, $states];
    }

    protected function differentiate(NDArray $dOutputs, array $dStates=null)
    {
        [$dWordvect,$dStates] = $this->rnn->backward($dOutputs,$dStates);
        $dInputs = $this->embedding->backward($dWordvect);
        return $dInputs;
    }

    public function initializeHiddenState($batch_sz)
    {
        return $this->backend->zeros([$batch_sz, $this->units]);
    }
}

class Decoder extends AbstractRNNLayer
{
    protected $backend;
    protected $vocabSize;
    protected $wordVectSize;
    protected $units;
    protected $targetLength;
    protected $embedding;
    protected $rnn;
    protected $attention;
    protected $concat;
    protected $dense;

    public function __construct(
        $backend,
        $builder,
        int $vocabSize,
        int $wordVectSize,
        int $units,
        int $targetLength
        )
    {
        $this->backend = $backend;
        $this->vocabSize = $vocabSize;
        $this->wordVectSize = $wordVectSize;
        $this->units = $units;
        $this->targetLength = $targetLength;
        $this->embedding = $builder->layers()->Embedding($vocabSize, $wordVectSize);
        $this->rnn = $builder->layers()->GRU($units,
            ['return_state'=>true,'return_sequences'=>true]
        );
        $this->attention = $builder->layers()->Attention(
            ['return_attention_scores'=>true]);
        $this->concat = $builder->layers()->Concatenate();
        $this->dense = $builder->layers()->Dense($vocabSize);
    }

    public function build(array $inputShape=null, array $options=null) : array
    {
        $inputShape=$this->normalizeInputShape($inputShape);
        $inputShape = $this->registerLayer($this->embedding,$inputShape);
        $rnnShape = $this->registerLayer($this->rnn,$inputShape);
        $inputShape = $this->registerLayer($this->attention,
            [$inputShape,[$this->targetLength,$this->units]]);
        $inputShape = $this->registerLayer($this->concat,[$inputShape,$rnnShape]);
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
            'units'=>$this->units,
        ];
    }

    protected function call(
        NDArray $inputs,
        bool $training,
        array $initial_state=null,
        array $options=null
        ) : array
        $K = $this->backend;
        $encOutputs=$options['enc_outputs'];

        $x = $this->embedding->forward($inputs);
        $rnnSequence,$states = $this->rnn->forward(
            $x,$training,$initial_state);

        [$contextVector,$attentionScores] = $this->attention->forward(
            [$rnnSequence,$encOutputs]);
        $outputs = $this->concat->forward([$contextVector, $rnnSequence]);

        $outputs = $this->dense->forward($outputs);
        $this->contextVectorShape = $contextVector->shape();
        $this->rnnSequenceShape = $rnnSequence->shape();
        $this->attentionScores = $attentionScores;
        return [$outputs,$states];
    }

    public function getAttentionScores()
    {
        return $this->attentionScores;
    }

    protected function differentiate(NDArray $dOutputs, array $dNextStates=null)
    {
        $K = $this->backend;
        $dOutputs = $this->dense->backward($dOutputs);
        [$dContextVector,$dRnnSequence] = $this->concat->backward($dOutputs);
        [$dRnnSequence2,$dEncOutputs] = $this->attention->backward($contextVector);
        $K->update_add($dRnnSequence,$dRnnSequence2);
        [$dWordVect,$dStates]=$this->rnn->backward($dRnnSequence,$dNextStates);
        $dInputs = $this->embedding->backward($dWordVect);
        return [$dInputs,$dStates,['enc_outputs'=>$dEncOutputs]];
    }
}


class Seq2seq extends AbstractModel
{
    public function __construct(
        $inputLength=null,
        $inputVocabSize=null,
        $outputLength=null,
        $targetVocabSize=null,
        $wordVectSize=8,
        $units=256,
        $startVocId=0,
        $endVocId=0
        )
    {
        self::__construct($backend,$builder);
        $this->encoder = Encoder(
            $inputVocabSize,
            $wordVectSize,
            $units
        );
        $this->decoder = Decoder(
            $targetVocabSize,
            $wordVectSize,
            $units
        );
        $this->out = $builder->layers()->Activation('softmax');
        $this->setLastLayer($this->out);
        $this->startVocId = $startVocId;
        $this->endVocId = $endVocId;
        $this->inputLength = $inputLength;
        $this->outputLength = $outputLength;
        $this->units = $units;
    }

    protected function buildLayers(array $options=null) : void
    {
        $shape = $this->registerLayer($this->encoder);
        $shape = $this->registerLayer($this->decoder,$shape);
        $this->registerLayer($this->out,$shape);
    }

    public function shiftLeftSentence(
        NDArray $sentence,
        ) : NDArray
    {
        $shape = $sentence->shape();
        $batchs = $shape[0];
        $zeroPad = $K->zeros([$batchs,1,1],$sentence->dtype());
        $seq = $K->slice($sentence,[0,1],[-1,-1]);
        $result = $K->concat([$seq,$zeroPad],$axis=1);
        return $result;
    }

    protected function forwardStep(NDArray $inputs, NDArray $trues=null, bool $training=null) : NDArray
    {
        $K = $this->backend;
        [$encOutputs,$states] = $this->encoder->forward($inputs,$training);
        $options = ['enc_outputs'=>$encOutputs];
        [$outputs,$dummy] = $this->decoder->forward($trues,$training,$states,$options);
        $outputs = $this->out->forward($outputs,$training);
        return $outputs;
    }

    protected function loss(NDArray $trues,NDArray $preds) : float
    {
        $K = $this->backend;
        $trues = $this->shiftLeftSentence($trues);
        #$mask = $K->equal($trues, $K->zerosLike($trues));
        #$mask2 = $K->oneHot($trues,$preds->shape()[2]);

        $loss = $this->lossFunction->loss($trues,$preds);

        #mask = tf.cast(mask, dtype=loss_.dtype)

        #loss_ *= mask

        #return tf.reduce_mean(loss_)
        return $loss;
    }

    protected function backwardStep(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        $dOutputs = $this->out->backward($dOutputs);
        [$dummy,$dStates,$dOptions] = $this->decoder->backward($dOutputs,null);
        $dEncOutputs = $dOptions['enc_outputs'];
        [$dInputs,$dStates] = $this->encoder->backward($dEncOutputs,$dStates);
        return $dInputs;
    }

    public function predict(NDArray $inputs, array $options=null) : NDArray
    {
        $K = $this->backend;
        $attentionPlot = $options['attention_plot'];

        if($inputs->ndim()!=2) {
            throw new InvalidArgumentException('inputs shape must be 2D.');
        }
        $batchs = $inputs->shape()[0];
        if($batchs!=1) {
            throw new InvalidArgumentException('num of batch must be one.');
        }
        $status = [$K->zeros([$batchs, $this->units])];
        [$encOutputs, $status] = $this->encoder($inputs, $training=false, $status);

        $decInputs = $K->array([[$this->start_voc_id]],$inputs->dtype());

        $result = [];
        for($t=0;$t<$this->output_length;$t++) {
            [$predictions, $status] = $this->decoder->forward(
                $decInputs, $training=false, $status, $encOutputs);

            # storing the attention weights to plot later on
            $scores = $this->decoder->getAttentionScores();
            $K->copy($scores->reshape([$inputLength]),$attentionPlot[$t]);

            $predictedId = $K->argmax($predictions[0][0]);

            $result[] = $predictedId;

            if($this->endVocId == $predictedId):
                break;

            # the predicted ID is fed back into the model
            $decInputs = $K->array([[$predictedId]],$inputs->dtype());
        }
        $result = $K->array([$result]);
        #return result, sentence, attention_plot
        return $result;
    }

    public function plot_attention(
        $attention, $sentence, $predictedSentence)
    {
        $plt = $this->plt;
        $plt->figure();
        #attention = attention[:len(predicted_sentence), :len(sentence)]
        $plt->imshow($attention, $cmap='viridis');

        $plt->xlabel($sentence);
        $plt->ylabel($predictedSentence);
    }
}

$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);

$lossObject = $nn->losses->SparseCategoricalCrossentropy();

def loss_function(real, pred):

$numExamples=5000;
$numWords=128;

$dataset = new EngFraDataset($mo);
[$input_tensor, $target_tensor, $inp_lang, $targ_lang_tokenizer]
    = $dataset->loadData(null,$numExamples,$numWords);


/*
            #sentence = dataset.preprocess_sentence(sentence)

            ##inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
            #inputs = lang_tokenizer.texts_to_sequences([sentence])
            inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,
                                                             maxlen=self.input_length,
                                                             padding='post')
            inputs = tf.convert_to_tensor(inputs)
            $attention_plot = $thi->mo->zeros([self.output_length, self.input_length])

            inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,
                                                             maxlen=self.input_length,
                                                             padding='post')
            inputs = tf.convert_to_tensor(inputs)
*/
