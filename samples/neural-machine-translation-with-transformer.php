<?php
require __DIR__.'/../vendor/autoload.php';

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Layer\Layer;
use Rindow\NeuralNetworks\Model\AbstractModel;
use Rindow\NeuralNetworks\Model\Model;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Gradient\Module;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Data\Sequence\Tokenizer;
use Rindow\NeuralNetworks\Data\Sequence\Preprocessor;
use Rindow\NeuralNetworks\Optimizer\Schedule\LearningRateSchedule;
use function Rindow\Math\Matrix\R;

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
        if(!class_exists("ZipArchive")) {
            throw new \Exception("Please configure the zip php-extension.");
        }
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

class PositionalEmbedding extends AbstractModel
{
    protected int $d_model;
    protected object $gradient;
    protected Layer $embedding;
    protected Layer $inheritMask;
    protected Variable $posEncoding;

    public function __construct(
        object $builder,
        int $vocab_size,
        int $d_model,
        int $maximumPositionEncoding=null,
        int $inputLength=null,
        )
    {
        parent::__construct($builder,name:'posEmb');
        $this->d_model = $d_model;
        $maximumPositionEncoding ??= 256;
        $this->gradient = $builder->gradient();
        $nn = $builder;
        $g = $this->gradient;

        $this->embedding = $nn->layers->Embedding(
            $vocab_size,    // inputDim
            $d_model,       // outputDim
            mask_zero:true,
            input_length:$inputLength,
        );
        $this->inheritMask = $nn->layers->InheritMask();

        $this->posEncoding = $g->Variable(
            $this->positionalEncoding(maxLength:$maximumPositionEncoding, depth:$d_model),
            trainable:false,
            unbackpropagatable:true,
            name:'posEncoding',
        );
    }

    public function positionalEncoding(int $maxLength, int $depth) : NDArray
    {
        $K = $this->backend;
        if($depth%2 != 0) {
            throw new InvalidArgumentException("depth must be a multiple of 2");
        }
        $depth = $depth/2;

        $positions = $K->repeat(                                # (maxLength, depth/2)
            $this->range(0,$maxLength),$depth,axis:1);
        $depths = $K->scale(1/$depth,$this->range(0,$depth));   # (depth/2)
        $angleRates = $K->reciprocal($this->pow(10000,$depths));# (depth/2)
        $angleRads = $K->mul($positions,$angleRates);           # (maxLength, depth/2)
      
        $posEncoding = $K->concat(                              # (maxLength, depth/2*2)
            [$K->sin($angleRads), $K->cos($angleRads)],
            $axis=-1); 
      
        return $posEncoding;                                    # (maxLength, depth)
    }

    protected function pow(NDArray|float $x, NDArray $y)
    {
        $K = $this->backend;
        if(is_numeric($x)) {
            $x = $K->fill($y->shape(),$x,dtype:$y->dtype());
        }
        //return $K->exp($K->mul($y,$K->log($x)));
        return $K->pow($x,$y);
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
            throw new InvalidArgumentException(
                "range has invalid args: start=$start,limit=$limit,delta=$delta");
        }
        if($dtype===null) {
            $dtype = NDArray::float32;
        }
        $count = (int)ceil(($limit-$start)/$delta);
        $y = $K->alloc([$count],$dtype);
        $d = $K->fill([1],$delta,$dtype);
        for($i=0; $i<$count; $i++) {
            $K->update($y[R($i,$i+1)],$K->scale($i,$d));
        }
        return $y;
    }

    public function call(NDArray $inputs)
    {
        $g = $this->gradient;
        // Embedding
        $input_vect = $this->embedding->forward($inputs);

        // positional Encoding
        $inputShape = $g->shape($inputs);
        $length = $g->get($inputShape,1);
        $x = $g->scale(sqrt($this->d_model), $input_vect);
        $pos_encoding = $g->get($this->posEncoding,0,$length);
        $x = $g->add($x, $pos_encoding); // broadcast add
        $x = $this->inheritMask->forward([$x,$input_vect]);
        return $x;
    }

    public function computeMask(...$args)
    {
        return $this->embedding->computeMask(...$args);
    }

}


abstract class AbstractBaseAttention extends AbstractModel
{
    protected Layer $mha;
    protected Layer $layernorm;
    protected Layer $add;
    
    public function __construct(
        object $nn,
        string $name=null,
        ...$args
        )
    {
        parent::__construct($nn,name:$name);
        $this->mha = $nn->layers->MultiHeadAttention(...$args);
        $this->layernorm = $nn->layers->LayerNormalization();
        $this->add = $nn->layers->Add();
    }
}

class CrossAttention extends AbstractBaseAttention
{
    public ?NDArray $last_attn_scores=null;

    protected function call(NDArray $x, NDArray $context, Variable|bool $training=null)
    {
        [$attn_output, $attn_scores] = $this->mha->forward(
            [
                $x, // query
                $context, // value
                $context, // key
            ],
            training:$training,
            returnAttentionScores:True
        );
    
        # Cache the attention scores for plotting later.
        $this->last_attn_scores = $this->builder->gradient()->ndarray($attn_scores);
    
        $x = $this->add->forward([$x, $attn_output]);
        $x = $this->layernorm->forward($x);
        return $x;
    }
}

class GlobalSelfAttention extends AbstractBaseAttention
{
    protected function call(NDArray $x, Variable|bool $training=null)
    {
        $attn_output = $this->mha->forward(
            [
                $x,  // query=
                $x,  // value=
                $x   // key=
            ],
            training:$training,
        );
        $x = $this->add->forward([$x, $attn_output]);
        $x = $this->layernorm->forward($x);
        return $x;
    }
}

class CausalSelfAttention  extends AbstractBaseAttention
{
    protected function call(NDArray $x, Variable|bool $training=null)
    {
        $attn_output = $this->mha->forward(
            [
                $x, // query
                $x, // value
                $x, // key                
            ],
            training:$training,
            useCausalMask:True,
        );
        $x = $this->add->forward([$x, $attn_output]);
        //echo 'layernorm IN=('.implode(',',$x->shape()).")\n";
        $x = $this->layernorm->forward($x);
        return $x;
    }
}

class FeedForward extends AbstractModel
{
    protected Model $seq;
    protected Layer $add;
    protected Layer $layer_norm;

    public function __construct(
        object $nn,
        int $d_model,
        int $dff,
        float $dropout_rate=0.1,
        string $name=null,
    ) {
        parent::__construct($nn,name:$name);
        $this->seq = $nn->models->Sequential([
            $nn->layers->Dense($dff, activation:'relu'),
            $nn->layers->Dense($d_model),
            $nn->layers->Dropout($dropout_rate),
        ]);
        $this->add = $nn->layers->Add();
        $this->layer_norm = $nn->layers->LayerNormalization();
    }

    protected function call(NDArray $x, Variable|bool $training=null)
    {
        $x = $this->add->forward([$x, $this->seq->forward($x,training:$training)]);
        $x = $this->layer_norm->forward($x);
        return $x;
    }

}

class EncoderLayer extends AbstractModel
{
    protected Model $self_attention;
    protected Model $ffn;

    public function __construct(
        object $nn,
        int $d_model,
        int $num_heads,
        int $dff,
        float $dropout_rate=0.1,
    ) {
        parent::__construct($nn,name:'encoderlayer');
        $this->self_attention = new GlobalSelfAttention(
            $nn,
            num_heads:$num_heads,
            key_dim:$d_model,
            dropout:$dropout_rate,
            name:'globalattn.encoderlayer'
        );
        $this->ffn = new FeedForward($nn, $d_model, $dff, name:'ff.encoderlayer');
    }

    protected function call(NDArray $x, Variable|bool $training=null)
    {
        $x = $this->self_attention->forward($x,training:$training);
        $x = $this->ffn->forward($x,training:$training);
        return $x;
    }
}

class Encoder extends AbstractModel
{
    protected int $d_model;
    protected int $num_layers;
    protected Model $pos_embedding;
    protected Module $enc_layers;
    protected Layer $dropout;

    public function __construct(
        object $nn,
        int $num_layers,
        int $d_model,
        int $num_heads,
        int $dff,
        int $vocab_size,
        float $dropout_rate=0.1,
    ) {
        parent::__construct($nn,name:'encoder');
        $this->d_model = $d_model;
        $this->num_layers = $num_layers;
    
        $this->pos_embedding = new PositionalEmbedding(
            $nn,
            vocab_size:$vocab_size,
            d_model:$d_model
        );
    
        $this->enc_layers = $nn->gradient->modules();
        for($i=0;$i<$num_layers;$i++) {
            $this->enc_layers->add(new EncoderLayer(
                $nn,
                d_model:$d_model,
                num_heads:$num_heads,
                dff:$dff,
                dropout_rate:$dropout_rate,
            ));
        }
        $this->dropout = $nn->layers->Dropout($dropout_rate);
    }

    protected function call(NDArray $x, Variable|bool $training=null)
    {
        # `x` is token-IDs shape: (batch, seq_len)
        $x = $this->pos_embedding->forward($x);  # Shape `(batch_size, seq_len, d_model)`.

        # Add dropout.
        $x = $this->dropout->forward($x, training:$training);

        foreach($this->enc_layers as $enc_layer) {
            $x = $enc_layer->forward($x, training:$training);
        }

        return $x;  # Shape `(batch_size, seq_len, d_model)`.
    }
}

class DecoderLayer extends AbstractModel
{
    protected Model $causal_self_attention;
    protected Model $cross_attention;
    protected Model $ffn;
    public ?NDArray $last_attn_scores=null;

    public function __construct(
        object $nn,
        int $d_model,
        int $num_heads,
        int $dff,
        float $dropout_rate=0.1,
    ) {
        parent::__construct($nn);
        $this->causal_self_attention = new CausalSelfAttention(
            $nn,
            num_heads:$num_heads,
            key_dim:$d_model,
            dropout:$dropout_rate,
            name:'causalatten.decoderlayer',
        );
    
        $this->cross_attention = new CrossAttention(
            $nn,
            num_heads:$num_heads,
            key_dim:$d_model,
            dropout:$dropout_rate,
            name:'crossAttn.decoderlayer',
        );
    
        $this->ffn = new FeedForward(
            $nn,
            $d_model,
            $dff,
            name:'ff.decoderlayer',
        );
    
    }

    protected function call(NDArray $x, NDArray $context, Variable|bool $training=null)
    {
        $x = $this->causal_self_attention->forward($x, training:$training);
        $x = $this->cross_attention->forward($x, $context, training:$training);
    
        # Cache the last attention scores for plotting later
        $this->last_attn_scores = $this->cross_attention->last_attn_scores;
    
        $x = $this->ffn->forward($x, training:$training);  # Shape `(batch_size, seq_len, d_model)`.
        return $x;
    }
}

class Decoder extends AbstractModel
{
    protected int $d_model;
    protected int $num_layers;
    protected Model $pos_embedding;
    protected Layer $dropout;
    protected Module $dec_layers;
    public ?NDArray $last_attn_scores=null;

    public function __construct(
        object $nn,
        int $num_layers,
        int $d_model,
        int $num_heads,
        int $dff,
        int $vocab_size,
        float $dropout_rate=0.1,
    ) {
        parent::__construct($nn,name:'decoder');
        $this->d_model = $d_model;
        $this->num_layers = $num_layers;
    
        $this->pos_embedding = new PositionalEmbedding(
            $nn,
            vocab_size:$vocab_size,
            d_model:$d_model,
        );
        $this->dropout = $nn->layers->Dropout($dropout_rate);
        $this->dec_layers = $nn->gradient->modules();
        for($i=0;$i<$num_layers;$i++) {
            $this->dec_layers->add(new DecoderLayer(
                $nn,
                d_model:$d_model,
                num_heads:$num_heads,
                dff:$dff,
                dropout_rate:$dropout_rate,
            ));
        }

        $this->last_attn_scores = null;
    }

    protected function call(NDArray $x, NDArray $context, Variable|bool $training=null)
    {
        # `x` is token-IDs shape (batch, target_seq_len)
        $x = $this->pos_embedding->forward($x);  # (batch_size, target_seq_len, d_model)

        $x = $this->dropout->forward($x, $training);

        foreach($this->dec_layers as $dec_layer) {
            $x = $dec_layer->forward($x, $context, training:$training);
        }

        $this->last_attn_scores = $dec_layer->last_attn_scores;

        # The shape of x is (batch_size, target_seq_len, d_model).
        return $x;
    }
}


class Transformer extends AbstractModel
{
    protected Model $encoder;
    public Model $decoder;
    protected Layer $final_layer;

    public function __construct(
        object $nn,
        int $num_layers,
        int $d_model,
        int $num_heads,
        int $dff,
        int $input_vocab_size,
        int $target_vocab_size,
        float $dropout_rate=0.1,
    )
    {
        parent::__construct($nn,name:'transformer');
        $this->encoder = new Encoder(
            $nn,
            num_layers:$num_layers,
            d_model:$d_model,
            num_heads:$num_heads,
            dff:$dff,
            vocab_size:$input_vocab_size,
            dropout_rate:$dropout_rate,
        );

        $this->decoder = new Decoder(
            $nn,
            num_layers:$num_layers,
            d_model:$d_model,
            num_heads:$num_heads,
            dff:$dff,
            vocab_size:$target_vocab_size,
            dropout_rate:$dropout_rate,
        );

        $this->final_layer = $nn->layers->Dense($target_vocab_size);

    }

    protected function call(NDArray $context, NDArray $x, Variable|bool $training=null) : NDArray
    {
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        # [$context, $x]  = $inputs;
        
        $context = $this->encoder->forward($context,training:$training);  # (batch_size, context_len, d_model)
        
        $x = $this->decoder->forward($x, $context,training:$training);  # (batch_size, target_len, d_model)
        
        # Final linear layer output.
        $logits = $this->final_layer->forward($x);  # (batch_size, target_len, target_vocab_size)
        
        #try:
        #    # Drop the keras mask, so it doesn't scale the losses/metrics.
        #    # b/250038731
        #    del logits._keras_mask
        #except AttributeError:
        #    pass
        
        # Return the final output and the attention weights.
        return $logits;
    }
}


class Translator
{
    protected object $builder;
    protected Model $transformer;
    protected int $max_out_length;
    protected int $start_voc_id;
    protected int $end_voc_id;

    public function __construct(
        object $nn,
        Model $transformer,
        int $max_out_length=null,
        int $start_voc_id=null,
        int $end_voc_id=null,
    )
    {
        $this->builder = $nn;
        $this->transformer = $transformer;
        $this->max_out_length = $max_out_length;
        $this->start_voc_id = $start_voc_id;
        $this->end_voc_id = $end_voc_id;
    }

    public function predict(NDArray $sentence) : array
    {
        $g = $this->builder->gradient();
        $K = $this->builder->backend();
        $sentence = $K->array($sentence);

        # The input sentence is Portuguese, hence adding the `[START]` and `[END]` tokens.
        #if len(sentence.shape) == 0:
        #  sentence = sentence[tf.newaxis]
        #
        #sentence = self.tokenizers.pt.tokenize(sentence).to_tensor()

        $encoder_input = $K->expandDims($sentence, axis:0);

        # As the output language is English, initialize the output with the
        # English `[START]` token.
        #start_end = self.tokenizers.en.tokenize([''])[0]
        #start = start_end[0][tf.newaxis]
        #end = start_end[1][tf.newaxis]
        $start = $K->array([[$this->start_voc_id]],dtype:NDArray::int32);
        $end   = $K->array([[$this->end_voc_id]],dtype:NDArray::int32);

        # `tf.TensorArray` is required here (instead of a Python list), so that the
        # dynamic-loop can be traced by `tf.function`.
        $output_array = $K->zeros([$this->max_out_length,1],dtype:NDArray::int32);
        $K->copy($start,$output_array[R(0,1)]);

        $this->transformer->setShapeInspection(false);
        for($i=0;$i<$this->max_out_length-1;$i++) {
            $output = $g->Variable($output_array[R(0,$i+1)]->reshape([1,$i+1]));
            $predictions = $this->transformer->forward($encoder_input, $output, training:false);

            # Select the last token from the `seq_len` dimension.
            #$predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.
            $predictions = $K->slice($predictions,[0, -1],[count($predictions), 1]);

            $predicted_id = $K->argmax($predictions, axis:-1);

            # Concatenate the `predicted_id` to the output which is given to the
            # decoder as its input.
            $K->copy($predicted_id[0]->reshape([1,1]),$output_array[R($i+1,$i+2)]);

            if($predicted_id[0]->toArray() == $end[0]->toArray()) {
                break;
            }
        }
        $output_len = $i+1;
        $output = $output_array[R(0,$output_len+1)];
        
        # The output shape is `(1, tokens)`.
        #text = tokenizers.en.detokenize(output)[0]  # Shape: `()`.

        #tokens = tokenizers.en.lookup(output)[0]

        # `tf.function` prevents us from using the attention_weights that were
        # calculated on the last iteration of the loop.
        # So, recalculate them outside the loop.
        $this->transformer->forward(
            $encoder_input,
            $output[R(0,$output_len)]->reshape([1,$output_len]),
            training:False
        );
        $this->transformer->setShapeInspection(true);
        $attention_weights = $this->transformer->decoder->last_attn_scores;

        $output = $output->reshape([1,$output_len+1]);
        #return text, tokens, attention_weights
        $output = $K->ndarray($output);
        $attention_weights = $K->ndarray($attention_weights);
        return [$output,$attention_weights];

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


class CustomSchedule implements LearningRateSchedule
{
    protected float $d_model;
    protected float $warmup_steps;

    public function __construct(int $d_model, int $warmup_steps=4000)
    {
        $this->d_model = (float)$d_model;
        $this->warmup_steps = (float)$warmup_steps;
    }

    public function __invoke(mixed $step) : float
    {
        $arg1 = 1 / sqrt($step);
        $arg2 = $step * ($this->warmup_steps ** -1.5);
        $lr = 1 / sqrt($this->d_model) * min($arg1, $arg2);
        return $lr;
    }
}

class CustomLossFunction
{
    protected $loss_object;
    protected $gradient;

    public function __construct($nn)
    {
        $this->gradient = $nn->gradient();
        $this->loss_object = $nn->losses->SparseCategoricalCrossentropy(
            from_logits:true, reduction:'none');
    }

    public function __invoke(NDArray $label, NDArray $pred) : NDArray
    {
        $g = $this->gradient;
        $loss = $this->loss_object->forward($label, $pred);
        $mask = $g->greater($g->cast($label,dtype:NDArray::float32),0.0);
        $loss = $g->mul($loss,$mask);
        return $g->div($g->reduceSum($loss),$g->reduceSum($mask));
    }
}

//class CustomAccuracy
//{
//    protected $backend;
//    protected $nn;
//    protected $gradient;
//
//    public function __construct($nn)
//    {
//        $this->backend = $nn->backend();
//        $this->nn = $nn;
//        $this->gradient = $nn->gradient();
//    }
//
//    public function __invoke($label, $pred)
//    {
//        $K = $this->backend;
//        $pred = $K->argMax($pred, axis:2);
//        $label = $K->cast($label,dtype:$pred->dtype());
//        $match = $K->equal($label, $pred);
//        $mask = $K->notEqual($label, $K->zerosLike($label));
//        $match = $K->cast($match,dtype:NDArray::float32);
//        $mask = $K->cast($mask,dtype:NDArray::float32);
//        $match = $K->mul($match,$mask);
//        $sumMatch = $K->sum($match);
//        if(is_numeric($sumMatch)) {
//            $accuracy = $sumMatch/$K->sum($mask);
//        } else {
//            $accuracy = $K->add($sumMatch,$K->sum($mask));
//            $accuracy = $K->scalar($accuracy);
//        }
//        return $accuracy;
//    }
//}
class CustomAccuracy
{
    protected $backend;
    protected $nn;
    protected $gradient;

    public function __construct($nn)
    {
        $this->backend = $nn->backend();
        $this->nn = $nn;
    }

    public function __invoke($label, $pred)
    {
        $mo = $this->nn->backend()->localMatrixOperator();
        $K = $this->backend;
        $pred = $K->argMax($pred, axis:-1);  // convert to token id from predicts

        $match = $K->equal($label, $pred);   // compare to trues (int32 == int32) 
        $mask = $K->cast($label,dtype:NDArray::bool); // make mask
        $match = $K->cast($match,dtype:NDArray::float32);
        $match = $K->masking($mask,$match); // masking matching results

        //echo "match=".$mo->shapeToString($match->shape())."\n";
        //echo "mask=".$mo->shapeToString($mask->shape())."\n";
        //echo "match=".$mo->toString($match,indent:true)."\n";
        //echo "mask=".$mo->toString($mask,indent:true)."\n";
        $sumMatch = $K->scalar($K->sum($match));
        $n = $K->scalar($K->sum($mask));
        if($n==0) {
            $accuracy = 0;
        } else {
            $accuracy = $sumMatch/$n;
        }
        return $accuracy;
    }
}

function make_labels($la,$label_tensor) {
    [$lebel_len,$lebel_words] = $label_tensor->shape();
    //$label_tensor  = $label_tensor,[:,1:$lebel_words];
    $label_tensor = $la->slice(
        $label_tensor,
        $start=[0,1],
        $size=[-1,$lebel_words-1]
        );
    $filler = $la->zeros($la->alloc([$lebel_len],dtype:$label_tensor->dtype()));
    $filler = $filler->reshape([$lebel_len,1]);
    //$label_tensor  = np.append($label_tensor,$filler,axis:1);
    $label_tensor = $la->concat(
        [$label_tensor,$filler],
        axis:1
        );
    return $label_tensor;
}

$numExamples=20000;#30000;#2000;
$numWords=1024;#null;
$epochs = 10;#20;
$batchSize = 64;#8;
$d_model=128;#256;#64;#128  // d_model embedding_dim
$dff=512;#64;  // units 
$num_layers=4;
$num_heads =8;

$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);
//$nn->backend()->primaryLA()->setProfiling(true);
$g = $nn->gradient();
$pltConfig = [];
$plt = new Plot($pltConfig,$mo);

$dataset = new EngFraDataset($mo);


echo "Generating data...\n";
[$inputTensor, $targetTensor, $inpLang, $targLang]
    = $dataset->loadData(null,$numExamples,$numWords);
$valSize = intval(floor(count($inputTensor)/100));
$trainSize = count($inputTensor)-$valSize;
//$inputTensorTrain  = $inputTensor[R(0,$trainSize)];
//$targetTensorTrain = $targetTensor[R(0,$trainSize)];
//$inputTensorVal  = $inputTensor[R($trainSize,$valSize+$trainSize)];
//$targetTensorVal = $targetTensor[R($trainSize,$valSize+$trainSize)];
$inputTensorTrain  = $inputTensor[R(0,$trainSize)];
$targetTensorTrain = $targetTensor[R(0,$trainSize)];
$inputTensorVal  = $inputTensor[R($trainSize,$valSize+$trainSize)];
$targetTensorVal = $targetTensor[R($trainSize,$valSize+$trainSize)];

$labelTensorTrain = make_labels($mo->la(),$targetTensorTrain);
$labelTensorVal = make_labels($mo->la(),$targetTensorVal);

$inputLength  = $inputTensor->shape()[1];
$outputLength = $targetTensor->shape()[1];
$inputVocabSize = $inpLang->numWords();
$targetVocabSize = $targLang->numWords();
$corpusSize = count($inputTensor);

echo "num_examples: $numExamples\n";
echo "num_words: $numWords\n";
echo "epoch: $epochs\n";
echo "batchSize: $batchSize\n";
echo "embedding_dim: $d_model\n";
echo "num_heads: $num_heads\n";
echo "dff: $dff\n";
echo "Total questions: $corpusSize\n";
echo "Input  word dictionary: $inputVocabSize(".$inpLang->numWords(true).")\n";
echo "Target word dictionary: $targetVocabSize(".$targLang->numWords(true).")\n";
echo "Input length: $inputLength\n";
echo "Output length: $outputLength\n";

$dataset = $nn->data->NDArrayDataset(
    [$inputTensorTrain,$targetTensorTrain],
    tests:  $labelTensorTrain, batch_size: 1, shuffle: false
);

echo "device type: ".$nn->deviceType()."\n";
$transformer = new Transformer(
    $nn,
    $num_layers,
    $d_model,      // d_model,
    $num_heads,
    $dff,
    $inputVocabSize,    // input_vocab_size,
    $targetVocabSize,   // target_vocab_size,
    $inputLength,       // inputLength,
    $outputLength,      // targetLength,
    // int $max_pe_input=null,
    // int $max_pe_target=null,
    // float $dropout_rate=0.1,
);

$lossfunc = new CustomLossFunction($nn);
$accuracyFunc = new CustomAccuracy($nn);
$learning_rate = new CustomSchedule($d_model);
$optimizer = $nn->optimizers->Adam(lr:$learning_rate, beta1:0.9, beta2:0.98,
                                     epsilon:1e-9);

echo "Compile model...\n";
$transformer->compile(
    loss:$lossfunc,
    optimizer:$optimizer,
    metrics:['loss'=>'loss','accuracy'=>$accuracyFunc],
);

$transformer->build(
    $g->ArraySpec([1,$inputLength],dtype:NDArray::int32),
    $g->ArraySpec([1,$outputLength],dtype:NDArray::int32),
    //trues:$g->ArraySpec([1,$outputLength],dtype:NDArray::int32)
); // just for summary
$transformer->summary();

$modelFilePath = __DIR__."/neural-machine-translation-with-transformer.model";

if(file_exists($modelFilePath)) {
    echo "Loading model...\n";
    $transformer->loadWeightsFromFile($modelFilePath);
} else {
    echo "Train model...\n";
    $start_time = time();
    $history = $transformer->fit(
        [$inputTensorTrain,$targetTensorTrain],
        $labelTensorTrain,
        batch_size:$batchSize,
        epochs:$epochs,
        //validation_data:[[$inputTensorVal,$targetTensorVal],$labelTensorVal],
        #callbacks:[checkpoint],
    );
    $trainint_time = time() - $start_time;
    echo "Total training time: ".gmdate('H:i:s', $trainint_time)."\n";
    $transformer->saveWeightsToFile($modelFilePath);

    $plt->figure();
    //$plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
    //$plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
    $plt->plot($mo->array($history['loss']),null,null,'loss');
    //$plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
    $plt->legend();
    $plt->title('seq2seq-transformer-translation');
}

$translator = new Translator(
    $nn,
    $transformer,
    max_out_length:$outputLength,
    start_voc_id:$targLang->wordToIndex('<start>'),
    end_voc_id:$targLang->wordToIndex('<end>'),
);

//$choice = $mo->random()->choice($corpusSize,10,false);
$choice = $mo->random()->choice($trainSize,10,false);
try {
foreach($choice as $idx)
{
    $question = $inputTensor[$idx];
    [$predict,$attentionPlot] = $translator->predict($question);
    $answer = $targetTensor[$idx];
    //$predictLabel = $mo->la()->reduceArgMax($mo->la()->squeeze($transformer->predict(inputs:[
    //    $mo->la()->expandDims($question,axis:0),$mo->la()->expandDims($answer,axis:0)])),axis:0);
        
    $sentence = $inpLang->sequencesToTexts([$question])[0];
    $predictedSentence = $targLang->sequencesToTexts($predict)[0];
    $targetSentence = $targLang->sequencesToTexts([$answer])[0];
    //$predictLabelSentence = $targLang->sequencesToTexts([$predictLabel])[0];
    echo "Input:   $sentence\n";
    echo "Predict: $predictedSentence\n";
    echo "Target:  $targetSentence\n";
    //echo "Label:  $predictLabelSentence\n";
    //echo "label:", $targLang->sequencesToTexts([$labelTensorTrain[$idx]])[0]."\n";
    echo "\n";    
}
} catch(\Exception $e) {
    echo get_class($e).": ".$e->getMessage()."\n";
    echo "File: ".$e->getFile()."(".$e->getLine().")\n";
    echo $e->getTraceAsString()."\n";
    echo "Exception!!!\n";
}
//$plt->show();
//$nn->backend()->primaryLA()->profilingReport();