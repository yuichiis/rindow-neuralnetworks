<?php
require __DIR__.'/../vendor/autoload.php';

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Layer\AbstractRNNLayer;
use Rindow\NeuralNetworks\Model\AbstractModel;
use Rindow\NeuralNetworks\Gradient\Variable;
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
    protected $mo;
    protected $datasetDir;
    protected $saveFile;
    protected $preprocessor;

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
            $tokenizer = new Tokenizer($this->mo,
                num_words: $numWords,
                filters: "\"\'#$%&()*+,-./:;=@[\\]^_`{|}~\t\n",
                specials: "?.!,¿",
            );
        }
        $tokenizer->fitOnTexts($lang);
        $sequences = $tokenizer->textsToSequences($lang);
        $tensor = $this->preprocessor->padSequences($sequences,padding:'post');
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

class MultiHeadAttention extends AbstractModel
{
    protected int $num_heads;
    protected int $depth;
    protected Layer $attention;
    protected Layer $wq;
    protected Layer $wk;
    protected Layer $wv;
    protected Layer $dense;
    protected object $gradient;

    public function __construct(
        object $backend,
        object $builder,
        int $depth,
        int $num_heads)
    {
        parent::__construct($backend,$builder);
        $this->gradient = $builder->gradient();
        $this->num_heads = $num_heads;
        $this->depth = $depth;
        if($depth % $num_heads != 0) {
            throw new InvalidArgumentException('"depth" must be an integer multiple of "num_heads"');
        }

        // self.depth = depth // self.num_heads
    
        $this->attention = $builder->layers->Attention(use_scale:true);
        $this->wq = $builder->layers->Dense($depth);
        $this->wk = $builder->layers->Dense($depth);
        $this->wv = $builder->layers->Dense($depth);
    
        $this->dense = $builder->layers->Dense($depth);
    }

    /**
     * Split the last dimension into (num_heads, depth).
     * Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
     */
    protected function split_heads($x)
    {
        $g = $this->gradient;
        $x = $g->reshape($x, [0, -1, $this->num_heads, $this->depth]);
        return $g->transpose($x, perm:[0, 2, 1, 3]);
    }

    protected function call($v, $k, $q, $mask)
    {
        $g = $this->gradient;

        $q = $this->wq->forward($q);  # (batch_size, seq_len, depth)
        $k = $this->wk->forward($k);  # (batch_size, seq_len, depth)
        $v = $this->wv->forward($v);  # (batch_size, seq_len, depth)
    
        $q = $this->split_heads($q);  # (batch_size, num_heads, seq_len_q, depth)
        $k = $this->split_heads($k);  # (batch_size, num_heads, seq_len_k, depth)
        $v = $this->split_heads($v);  # (batch_size, num_heads, seq_len_v, depth)
    
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        [$scaled_attention, $attention_weights] = 
            $this->attention->forward([$q, $k, $v], returnAttentionScores:true, mask:$mask);
    
        $scaled_attention = $g->transpose($scaled_attention, perm:[0, 2, 1, 3]);  # (batch_size, seq_len_q, num_heads, depth)
    
        $concat_attention = $g->reshape($scaled_attention,
                                      [0, -1, $this->depth]);  # (batch_size, seq_len_q, depth)
    
        $output = $this->dense($concat_attention);  # (batch_size, seq_len_q, depth)
    
        return [$output, $attention_weights];
    }
}

function point_wise_feed_forward_network(object $builder, int $depth, int $dff) : object
{
    return $builder->models->Sequential([
        $builder->layers->Dense($dff, activation:'relu'),   # (batch_size, seq_len, dff)
        $builder->layers->Dense($depth),                    # (batch_size, seq_len, depth)
    ]);
}

class EncoderLayer extends AbstractModel
{
    protected Model $mha;
    protected Model $fnn;
    protected Layer $layernorm1;
    protected Layer $layernorm2;

    public function __construct(
        object $backend,
        object $builder,
        int $wordVectSize=null, 
        int $num_heads=null, 
        int $dff=null, 
        float $dropout_rate=0.1)
    {
        parent::__construct($backend,$builder);
        $this->gradient = $builder->gradient();
        $this->mha = new MultiHeadAttention($wordVectSize, $num_heads);
        $this->ffn = point_wise_feed_forward_network($builder,$wordVectSize, $dff);
    
        $this->layernorm1 = $builder->layers->LayerNormalization(epsilon:1e-6);
        $this->layernorm2 = $builder->layers->LayerNormalization(epsilon:1e-6);
    
        $this->dropout1 = $builder->layers->Dropout($dropout_rate);
        $this->dropout2 = $builder->layers->Dropout($dropout_rate);
    }
  
    protected function call(
        NDArray $x, 
        Variable|bool $training, 
        NDArray $mask)
    {
        $g = $this->gradient;

        [$attn_output, $dummy] = $this->mha($x, $x, $x, $mask);  # (batch_size, input_seq_len, d_model)
        $attn_output = $this->dropout1($attn_output, $training);
        $out1 = $this->layernorm1($g->add([$x, $attn_output]));  # (batch_size, input_seq_len, d_model)
    
        $ffn_output = $this->ffn($out1);  # (batch_size, input_seq_len, d_model)
        $ffn_output = $this->dropout2($ffn_output, $training);
        $out2 = $this->layernorm2($g->add([$out1, $ffn_output]));  # (batch_size, input_seq_len, d_model)
        return $out2;
    }
}

class Encoder extends AbstractModel
{
    protected $backend;
    protected $gradient;
    protected $vocabSize;
    protected $wordVectSize;
    protected $units;
    protected $posEncoding;
    protected $embedding;
    protected $rnn;
    protected $mo;

    public function __construct(
        object $backend,
        object $builder,
        int $numLayers,
        int $num_heads,
        int $dff,
        int $vocabSize,
        int $wordVectSize,
        int $units,
        int $inputLength,
        float $dropout_rate,
        object $mo=null,
        )
    {
        parent::__construct($backend,$builder);
        $this->gradient = $builder->Gradient();
        $this->vocabSize = $vocabSize;
        $this->wordVectSize = $wordVectSize;
        $this->units = $units;
        $this->posEncoding = $this->positionalEncoding($length=256, $depth=$wordVectSize);
        $this->embedding = $builder->layers()->Embedding(
            $vocabSize,$wordVectSize,
            input_length:$inputLength
        );
        $this->enc_layers = $builder->models->SeqSequential();
        for($i=0;$i<$numLayers;$i++) {
            $this->enc_layers->add(
                new EncoderLayer(
                    $backend,$builder,
                    wordVectSize:$wordVectSize,
                    num_heads:$num_heads,
                    dff:$dff,
                    dropout_rate:$dropout_rate,
                )
            );
        }
        $this->dropout = $builder->layers->Dropout($dropout_rate);
    }

    protected function pow(NDArray|float $x, NDArray $y)
    {
        $K = $this->backend;
        if(is_numeric($x)) {
            $x = $K->fill($y->shape(),$x,$y->dtype());
        }
        return $K->exp($K->mul($y,$K->log($x)));
    }

    protected function range(float $start, float $limit, float $delta=null, $dtype=null) : NDArray
    {
        $K = $this->backend;
        if($delta===null) {
            if($start<$limit) {
                $delta = 1.0;
            } else {
                $delta = -1.0;
            }
        }
        if(($start==$limit)||($start<$limit && $delta<=0)||($start>$limit && $delta>=0)) {
            throw new InvalidArgumentException("range has invalid args: start=$start,limit=$limit,delta=$delta");
        }
        if($dtype===null) {
            $dtype = NDArray::float32;
        }
        $count = (int)ceil(($limit-$start)/$delta);
        $y = $K->alloc([$count],$dtype);
        $d = $K->fill([1],$delta,$dtype);
        for($i=0; $i<$count; $i++) {
            $K->update($y[[$i,$i]],$K->scale($i,$d));
        }
        return $y;
    }

    public function positionalEncoding(int $maxLength, int $depth) : NDArray
    {
        $K = $this->backend;
        if($depth%2 != 0) {
            throw new InvalidArgumentException("depth must be a multiple of 2");
        }
        $depth = $depth/2;

        $positions = $K->repeat(                                # [maxLength, depth]
            $this->range(0,$maxLength),$depth,axis:1);
        $depths = $K->scale(1/$depth,$this->range(0,$depth));   # [depth]
        $angleRates = $K->reciprocal($this->pow(10000,$depths));# [depth]
        $angleRads = $K->mul($positions,$angleRates);           # [maxLength, depth]
      
        $posEncoding = $K->concat(                              # [2 x maxLength, depth]
            [$K->sin($angleRads), $K->cos($angleRads)],
            $axis=-1); 
      
        return $posEncoding;
    }

    protected function call(
        object $inputs,
        Variable|bool $training,
        NDArray $mask=null
        ) : array
    {
        $g = $this->gradient;

        $inputShape = $g->shape($inputs);
        $length = $g->offsetGet($inputShape,1);

        $x = $this->embedding->forward($inputs,$training);

        // positional Encoding
        $x = $g->scale(sqrt($this->wordVectSize), $x);
        $pos_encoding = $g->slice($this->pos_encoding,[0],[$length]);
        $x = $g->add($x, $pos_encoding); // broadcast add
        # Add dropout.
        $x = $this->dropout->forward($x,$training);

        // seq layers
        $x = $this->enc_layers->forward($inputs,$training, $mask);

        return $x;  # Shape `(batch_size, seq_len, wordVectSize)`.
    }
}

class DecoderLayer extends AbstractModel
{
    protected Model $mha1;
    protected Model $mha2;
    protected Model $ffn;
    protected Layer $layernorm1;
    protected Layer $layernorm2;
    protected Layer $layernorm3;

    public function __construct(
        object $backend,
        object $builder,
        int $wordVectSize=null,
        int $num_heads=null,
        int $dff=null,
        float $dropout_rate=0.1
        )
    {
        parent::__construct($backend,$builder);
        $this->gradient = $builder->Gradient();
        $this->vocabSize = $vocabSize;
    
        $this->mha1 = new MultiHeadAttention($wordVectSize, $num_heads);
        $this->mha2 = new MultiHeadAttention($wordVectSize, $num_heads);
    
        $this->ffn = point_wise_feed_forward_network($wordVectSize, $dff);
    
        $this->layernorm1 = $builder->layers->LayerNormalization(epsilon:1e-6);
        $this->layernorm2 = $builder->layers->LayerNormalization(epsilon:1e-6);
        $this->layernorm3 = $builder->layers->LayerNormalization(epsilon:1e-6);
    
        $this->dropout1 = $builder->layers->Dropout($dropout_rate);
        $this->dropout2 = $builder->layers->Dropout($dropout_rate);
        $this->dropout3 = $builder->layers->Dropout($dropout_rate);
    }
    
    protected function call(
        NDArray $x,
        NDArray $enc_output,
        Variable|bool $training,
        NDArray $look_ahead_mask,
        NDArray $padding_mask)
    {
        # enc_output.shape == (batch_size, input_seq_len, d_model)
    
        [$attn1, $attn_weights_block1] = $this->mha1->forward($x, $x, $x, $look_ahead_mask);  # (batch_size, target_seq_len, wordVectSize)
        $attn1 = $this->dropout1->forward($attn1, $training);
        $out1 = $this->layernorm1->forward($g->add($attn1, + $x));
    
        [$attn2, $attn_weights_block2] = $this->mha2->forward(
            $enc_output, $enc_output, $out1, $padding_mask);  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training);
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
    
        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
    
        return out3, attn_weights_block1, attn_weights_block2
    }
}
  
class Decoder extends AbstractModel
{
    protected $backend;
    protected $vocabSize;
    protected $wordVectSize;
    protected $units;
    protected $inputLength;
    protected $targetLength;
    protected $embedding;
    protected $rnn;
    protected $attention;
    protected $concat;
    protected $dense;
    protected $attentionScores;

    public function __construct(
        $backend,
        $builder,
        int $vocabSize,
        int $wordVectSize,
        int $units,
        int $inputLength,
        int $targetLength
        )
    {
        $this->backend = $backend;
        $this->vocabSize = $vocabSize;
        $this->wordVectSize = $wordVectSize;
        $this->units = $units;
        $this->inputLength = $inputLength;
        $this->targetLength = $targetLength;
        $this->embedding = $builder->layers()->Embedding(
            $vocabSize, $wordVectSize,
            input_length:$targetLength
        );
        $this->rnn = $builder->layers()->GRU($units,
            return_state:true,return_sequences:true,
            recurrent_initializer:'glorot_uniform'
        );
        $this->attention = $builder->layers()->Attention();
        $this->concat = $builder->layers()->Concatenate();
        $this->dense = $builder->layers()->Dense($vocabSize);
    }

    protected function call(
        object $inputs,
        Variable|bool $training,
        array $initial_state=null,
        array $options=null
        ) : array
    {
        $K = $this->backend;
        $encOutputs=$options['enc_outputs'];

        $x = $this->embedding->forward($inputs,$training);
        [$rnnSequence,$states] = $this->rnn->forward(
            $x,$training,$initial_state);

        $contextVector = $this->attention->forward(
            [$rnnSequence,$encOutputs],$training,$options);
        if(is_array($contextVector)) {
            [$contextVector,$attentionScores] = $contextVector;
            $this->attentionScores = $attentionScores;
        }
        $outputs = $this->concat->forward([$contextVector, $rnnSequence],$training);

        $outputs = $this->dense->forward($outputs,$training);
        return [$outputs,$states];
    }

    public function getAttentionScores()
    {
        return $this->attentionScores;
    }
}


class Transformer extends AbstractModel
{
    protected $encoder;
    protected $decoder;
    protected $out;
    protected $mo;
    protected $backend;
    protected $startVocId;
    protected $endVocId;
    protected $inputLength;
    protected $outputLength;
    protected $units;
    protected $plt;

    public function __construct(
        $mo,
        $backend,
        $builder,
        $inputLength=null,
        $inputVocabSize=null,
        $outputLength=null,
        $targetVocabSize=null,
        $wordVectSize=8,
        $units=256,
        $startVocId=0,
        $endVocId=0,
        $plt=null
        )
    {
        parent::__construct($backend,$builder);
        $this->encoder = new Encoder(
            $backend,
            $builder,
            $inputVocabSize,
            $wordVectSize,
            $units,
            $inputLength
        );
        $this->decoder = new Decoder(
            $backend,
            $builder,
            $targetVocabSize,
            $wordVectSize,
            $units,
            $inputLength,
            $outputLength
        );
        $this->out = $builder->layers()->Activation('softmax');
        $this->mo = $mo;
        $this->backend = $backend;
        $this->startVocId = $startVocId;
        $this->endVocId = $endVocId;
        $this->inputLength = $inputLength;
        $this->outputLength = $outputLength;
        $this->units = $units;
        $this->plt = $plt;
    }

    def create_padding_mask(seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    def create_look_ahead_mask(size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)

    def create_masks(self, inp, tar):
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(inp)
    
        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = create_padding_mask(inp)
    
        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    
        return enc_padding_mask, look_ahead_mask, dec_padding_mask

    protected function call($inputs, $training=null $trues=null)
    {
        $K = $this->backend;
        [$encOutputs,$states] = $this->encoder->forward($inputs,$training);
        $options = ['enc_outputs'=>$encOutputs];
        [$outputs,$dmyStatus] = $this->decoder->forward($trues,$training,$states,$options);
        $outputs = $this->out->forward($outputs,$training);
        return $outputs;
    }

    public function shiftLeftSentence(
        NDArray $sentence
        ) : NDArray
    {
        $K = $this->backend;
        $shape = $sentence->shape();
        $batchs = $shape[0];
        $zeroPad = $K->zeros([$batchs,1],$sentence->dtype());
        $seq = $K->slice($sentence,[0,1],[-1,-1]);
        $result = $K->concat([$seq,$zeroPad],$axis=1);
        return $result;
    }

    protected function trueValuesFilter(NDArray $trues) : NDArray
    {
        return $this->shiftLeftSentence($trues);
    }

    public function predict($inputs, ...$options) : NDArray
    {
        $K = $this->backend;
        $attentionPlot = $options['attention_plot'];
        $inputs = $K->array($inputs);

        if($inputs->ndim()!=2) {
            throw new InvalidArgumentException('inputs shape must be 2D.');
        }
        $batchs = $inputs->shape()[0];
        if($batchs!=1) {
            throw new InvalidArgumentException('num of batch must be one.');
        }
        $status = [$K->zeros([$batchs, $this->units])];
        [$encOutputs, $status] = $this->encoder->forward($inputs, $training=false, $status);

        $decInputs = $K->array([[$this->startVocId]],$inputs->dtype());

        $result = [];
        $this->setShapeInspection(false);
        for($t=0;$t<$this->outputLength;$t++) {
            [$predictions, $status] = $this->decoder->forward(
                $decInputs, $training=false, $status,
                encOutputs:$encOutputs,returnAttentionScores:true);

            # storing the attention weights to plot later on
            $scores = $this->decoder->getAttentionScores();
            $this->mo->la()->copy(
                $K->ndarray($scores->reshape([$this->inputLength])),
                $attentionPlot[$t]);

            $predictedId = $K->scalar($K->argmax($predictions[0][0]));

            $result[] = $predictedId;

            if($this->endVocId == $predictedId) {
                $t++;
                break;
            }

            # the predicted ID is fed back into the model
            $decInputs = $K->array([[$predictedId]],$inputs->dtype());
        }

        $this->setShapeInspection(true);
        $result = $K->array([$result],NDArray::int32);
        return $K->ndarray($result);
    }

    public function plotAttention(
        $attention, $sentence, $predictedSentence)
    {
        $plt = $this->plt;
        $config = [
            'frame.xTickPosition'=>'up',
            'frame.xTickLabelAngle'=>90,
            'figure.topMargin'=>100,
        ];
        $plt->figure(null,null,$config);
        $sentenceLen = count($sentence);
        $predictLen = count($predictedSentence);
        $image = $this->mo->zeros([$predictLen,$sentenceLen],$attention->dtype());
        for($y=0;$y<$predictLen;$y++) {
            for($x=0;$x<$sentenceLen;$x++) {
                $image[$y][$x] = $attention[$y][$x];
            }
        }
        $plt->imshow($image, $cmap='viridis',null,null,$origin='upper');

        $plt->xticks($this->mo->arange(count($sentence)),$sentence);
        $predictedSentence = array_reverse($predictedSentence);
        $plt->yticks($this->mo->arange(count($predictedSentence)),$predictedSentence);
    }
}

$numExamples=20000;#30000
$numWords=1024;#null;
$epochs = 10;
$batchSize = 64;
$wordVectSize=256;  // d_model
$units=1024;


$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);
$pltConfig = [];
$plt = new Plot($pltConfig,$mo);

$dataset = new EngFraDataset($mo);

echo "Generating data...\n";
[$inputTensor, $targetTensor, $inpLang, $targLang]
    = $dataset->loadData(null,$numExamples,$numWords);
$valSize = intval(floor(count($inputTensor)/10));
$trainSize = count($inputTensor)-$valSize;
$inputTensorTrain  = $inputTensor[[0,$trainSize-1]];
$targetTensorTrain = $targetTensor[[0,$trainSize-1]];
$inputTensorVal  = $inputTensor[[$trainSize,$valSize+$trainSize-1]];
$targetTensorVal = $targetTensor[[$trainSize,$valSize+$trainSize-1]];

$inputLength  = $inputTensor->shape()[1];
$outputLength = $targetTensor->shape()[1];
$inputVocabSize = $inpLang->numWords();
$targetVocabSize = $targLang->numWords();
$corpusSize = count($inputTensor);

echo "num_examples: $numExamples\n";
echo "num_words: $numWords\n";
echo "epoch: $epochs\n";
echo "batchSize: $batchSize\n";
echo "embedding_dim: $wordVectSize\n";
echo "units: $units\n";
echo "Total questions: $corpusSize\n";
echo "Input  word dictionary: $inputVocabSize(".$inpLang->numWords(true).")\n";
echo "Target word dictionary: $targetVocabSize(".$targLang->numWords(true).")\n";
echo "Input length: $inputLength\n";
echo "Output length: $outputLength\n";

$tst = new Encoder(
        $nn->backend(),
        $nn,
        8,//$inputVocabSize,
        4,//$wordVectSize,
        8,//$units,
        4,//$inputLength,
        mo:$mo,
);
$g = $nn->Gradient();
$inputs = $g->Variable([[1,2,3,4]]);
[$y, $states] = $tst($inputs,true); 
echo $mo->toString($y,format:'%.7f',indent:true);
echo "end\n";
exit();

$seq2seq = new Seq2seq(
    $mo,
    $nn->backend(),
    $nn,
    $inputLength,
    $inputVocabSize,
    $outputLength,
    $targetVocabSize,
    $wordVectSize,
    $units,
    $targLang->wordToIndex('<start>'),
    $targLang->wordToIndex('<end>'),
    $plt
);

echo "Compile model...\n";
$seq2seq->compile(
    loss:'sparse_categorical_crossentropy',
    optimizer:'adam',
    metrics:['accuracy','loss'],
);
$seq2seq->build([1,$inputLength], trues:[1,$outputLength]); // just for summary
$seq2seq->summary();

$modelFilePath = __DIR__."/neural-machine-translation-with-attention.model";

if(file_exists($modelFilePath)) {
    echo "Loading model...\n";
    $seq2seq->loadWeightsFromFile($modelFilePath);
} else {
    echo "Train model...\n";
    $history = $seq2seq->fit(
        $inputTensorTrain,
        $targetTensorTrain,
            batch_size:$batchSize,
            epochs:$epochs,
            validation_data:[$inputTensorVal,$targetTensorVal],
            #callbacks:[checkpoint],
        );
    $seq2seq->saveWeightsToFile($modelFilePath);

    $plt->figure();
    $plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
    $plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
    $plt->plot($mo->array($history['loss']),null,null,'loss');
    $plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
    $plt->legend();
    $plt->title('seq2seq-attention-translation');
}

$choice = $mo->random()->choice($corpusSize,10,false);
foreach($choice as $idx)
{
    $question = $inputTensor[$idx]->reshape([1,$inputLength]);
    $attentionPlot = $mo->zeros([$outputLength, $inputLength]);
    $predict = $seq2seq->predict(
        $question,attention_plot:$attentionPlot);
    $answer = $targetTensor[$idx]->reshape([1,$outputLength]);;
    $sentence = $inpLang->sequencesToTexts($question)[0];
    $predictedSentence = $targLang->sequencesToTexts($predict)[0];
    $targetSentence = $targLang->sequencesToTexts($answer)[0];
    echo "Input:   $sentence\n";
    echo "Predict: $predictedSentence\n";
    echo "Target:  $targetSentence\n";
    echo "\n";
    $q = [];
    foreach($question[0] as $n) {
        if($n==0)
            break;
        $q[] = $inpLang->indexToWord($n);
    }
    $p = [];
    foreach($predict[0] as $n) {
        if($n==0)
            break;
        $p[] = $targLang->indexToWord($n);
    }
    $seq2seq->plotAttention($attentionPlot,  $q, $p);
}
$plt->show();
