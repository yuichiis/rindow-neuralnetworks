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
use Rindow\NeuralNetworks\Preprocessing\Sequence\Tokenizer;
use Rindow\NeuralNetworks\Preprocessing\Sequence\Preprocessor;

# Download the file
class EngFraDataset
{
    protected $baseUrl = 'http://www.manythings.org/anki/';
    protected $downloadFile = 'spa-eng.zip';

    public function __construct($mo,$inputTokenizer=null,$targetTokenizer=null)
    {
        $this->matrixOperator = $mo;
        $this->datasetDir = $this->getDatasetDir();
        if(!file_exists($this->datasetDir)) {
            @mkdir($this->datasetDir,0777,true);
        }
        $this->saveFile = $this->datasetDir . "/spa-eng.pkl";
        $this->preprocessor = new Preprocessor($mo);
    }

    protected function getDatasetDir()
    {
        return sys_get_temp_dir().'/rindow/nn/datasets/spa-eng';
    }

    protected function download($filename)
    {
        $filePath = $this->datasetDir . "/" . $filename;

        if(!file_exists($filePath)){
            $this->console("Downloading " . $filename . " ... ");
            copy($this->baseUrl.$filename, $filePath);
            $this->console("Done\n");
        }

        $path = $this->datasetDir.'/spa.txt';
        if(file_exists($path)){
            return $path;
        }
        $this->console("Extract to:".$this->datasetDir.'/..'."\n");
        $files = [];
        foreach($this->keyFiles as $file){
            $files[]='spa-eng/'.$file;
        }
        $phar = new PharData($filePath);
        $rc=$phar->extractTo($this->datasetDir.'/..',$files,true);
        $this->console("Done\n");

        return $path;
    }

    # Converts the unicode file to ascii
    #def unicode_to_ascii(self,s):
    #    return ''.join(c for c in unicodedata.normalize('NFD', s)
    #    if unicodedata.category(c) != 'Mn')

    public function preprocessSentence($w)
    {
        #w = self.unicode_to_ascii(w.lower().strip())
        $w = mb_convert_case($w, MB_CASE_LOWER);
        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
        # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
        $w = mb_ereg_replace("([?.!,¿])", " \1 ", $w);
        $w = mb_ereg_replace('[" "]+', " ", $w);
        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        $w = mb_ereg_replace("[^a-zA-Z?.!,¿]+", " ", $w);
        $w = trim($w);
        # adding a start and an end token to the sentence
        # so that the model know when to start and stop predicting.
        $w = '<start> '.$w.' <end>';
        return $w;
    }

    # 1. Remove the accents
    # 2. Clean the sentences
    # 3. Return word pairs in the format: [ENGLISH, SPANISH]
    public function createDataset($path, $numExamples)
    {
        $contents = file_get_contents($path);
        if($contents==false) {
            throw new InvalidArgumentException('file not found: '.$path);
        }
        $lines = explode('\n',trim($contents));
        unset($contents);
        $trim = function($w) { return trim($w); };
        $enSentences = [];
        $spSentences = [];
        foreach ($lines as $line) {
            $numExamples--;
            if($numExamples<0)
                break;
            $blocks = explode('\t',$line);
            $blocks = array_map($trim,$blocks);
            $en = $this->preprocessSentence($blocks[0]);
            $sp = $this->preprocessSentence($blocks[1]);
            $enSentences[] = $en;
            $spSentences[] = $sp;
        }
        return [$enSentences,$spSentences];
    }

    public function tokenize(string $lang,$tokenizer=null)
    {
        if($tokenizer==null) {
            $tokenizer = new Tokenizer();
        }
        $tokenizer->fitOnTexts($lang);
        $tensor = $tokenizer->textsToSequences($lang);
        $tensor = $this->preprocessor->padSequences($tensor,$padding='post');
        return $tensor, $tokenizer;
    }

    public function loadData(string $path=null, int $num_examples=null)
    {
        if($path==null) {
            $path = $this->download()
        }
        # creating cleaned input, output pairs
        [$targ_lang, $inp_lang] = $this->createDataset($path, $num_examples)

        [$input_tensor, $inp_lang_tokenizer] = $this->tokenize($inp_lang);
        [$target_tensor, $targ_lang_tokenizer] = $this->tokenize($targ_lang);
        $numInput = $input_tensor->shape()[0];
        $choice = $this->mo->random()->choice($numInput,$numInput,$replace=false);
        $input_tensor = $this->shuffle($input_tensor,$choice);
        $target_tensor = $this->shuffle($target_tensor,$choice);

        return [$input_tensor, $target_tensor, $inp_lang_tokenizer, $targ_lang_tokenizer]
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

    #def seq2str(self,sequence,lang):
    #    result = ''
    #    for word_id in sequence:
    #        if word_id == 0:
    #            break
    #        else:
    #            word = lang.index_word[word_id]
    #            if result=='':
    #                result = word
    #            else:
    #                result = result+' '+word
    #            if word == '<end>':
    #                return result
    #    return result

    public function seq2str(NDArray $sequence, $lang) : string
    {
        $result = '';
        $len = $sequence->shape()[0];
        for($i=0;$i<$len;$i++) {
            $word_id = $sequence[$i];
            if($word_id == 0) {
                $result .= ' ';
            } else {
                $word = $lang->index_word[$word_id];
                if($word == '<end>')
                    return $result;
                if($word != '<start>')
                    $result .= $word.' ';
            }
        }
        return $result;
    }
}

class Encoder(keras.Model):
    '''
    encoder
    '''
    def __init__(
        self,
        input_length: int,
        vocab_size: int,
        word_vect_size: int,
        recurrent_units: int,
        **kwargs
    ):
        '''encoder'''
        super(Encoder, self).__init__(**kwargs)
        #self.input_shape=[input_length]
        self.vocab_size = vocab_size
        self.word_vect_size = word_vect_size
        self.recurrent_units = recurrent_units

        self.embedding = keras.layers.Embedding(vocab_size, word_vect_size)
        self.rnn = keras.layers.GRU(recurrent_units,
            return_state=True,return_sequences=True,)
        #self.rnn = keras.layers.LSTM(recurrent_units,
        #    return_state=True,return_sequences=True,)
        #self.rnn = keras.layers.SimpleRNN(recurrent_units,
        #    return_state=True,return_sequences=True,)

    def call(
        self,
        inputs: ndarray,
        training: bool,
        initial_state: tuple=None,
        **kwargs) -> tuple:
        '''forward'''
        wordvect = self.embedding(inputs)
        outputs,states = self.rnn(wordvect,training=training,initial_state=initial_state)
        return (outputs, [states])

    def initialize_hidden_state(self,batch_sz):
        return tf.zeros((batch_sz, self.recurrent_units))

class Decoder(keras.Model):
    '''
    Decoder
    '''
    def __init__(
        self,
        input_length: int,
        vocab_size: int,
        word_vect_size: int,
        recurrent_units: int,
        **kwargs
        ):
        '''
        Decoder
        '''
        super(Decoder, self).__init__(**kwargs)
        #self.input_shape=[input_length]
        self.vocab_size = vocab_size
        self.word_vect_size = word_vect_size
        self.recurrent_size = recurrent_units

        self.embedding = keras.layers.Embedding(vocab_size, word_vect_size)
        self.rnn = keras.layers.GRU(recurrent_units,
            return_state=True,
            return_sequences=True,)
        #self.rnn = keras.layers.LSTM(recurrent_units,
        #    return_state=True,
        #    return_sequences=True,)
        #self.rnn = keras.layers.SimpleRNN(recurrent_units,
        #    return_state=True,
        #    return_sequences=True,)
        self.attention = keras.layers.Attention()
        #self.concat = keras.layers.Concatenate()
        self.dense = keras.layers.Dense(vocab_size)

    def call(
        self,
        inputs: ndarray,
        training: bool,
        initial_state: tuple=None,
        enc_outputs=None,
        **kwargs) -> tuple:
        '''forward'''

        x = self.embedding(inputs)
        #print('inputs in decoder:',inputs.shape)
        #print('inputs.embedding in decoder:',x.shape)
        #if initial_state is not None:
        #    print('initial_state in decoder:',initial_state[0].shape)
        #else:
        #    print('initial_state in decoder: None')
        rnn_sequence,states = self.rnn(x,training=training,initial_state=initial_state)

        #print('query(x) in decoder:',x.shape)
        #print('value(enc_outputs) in decoder:',enc_outputs.shape)
        context_vector = self.attention([rnn_sequence,enc_outputs])
        #print('context_vector in decoder:',context_vector.shape)
        x = tf.concat([context_vector, rnn_sequence], axis=-1)
        #print('concat in decoder:',x.shape)

        outputs = self.dense(x)
        return (outputs,[states],rnn_sequence)


class Seq2seq(keras.Model):

    def __init__(
        self,
        input_length=None,
        input_vocab_size=None,
        output_length=None,
        target_vocab_size=None,
        word_vect_size=8,
        recurrent_units=256,
        start_voc_id=0,
        **kwargs
    ):
        '''
        input_length: input sequence length
        input_vocab_size: vocabulary dictionary size for input sequence
        output_length: output sequence length
        target_vocab_size: vocabulary dictionary size for target sequence
        word_vect_size: word vector size of embedding layer
        recurrent_units: units of the recurrent layer
        dense_units: units of the full connection layer
        start_voc_id: vocabulary id of start word in input sequence
        '''
        super(Seq2seq, self).__init__(**kwargs)
        self.encoder = Encoder(
            input_length,
            input_vocab_size,
            word_vect_size,
            recurrent_units,
            **kwargs
        )
        self.decoder = Decoder(
            output_length,
            target_vocab_size,
            word_vect_size,
            recurrent_units,
            **kwargs
        )
        #self.out = keras.layers.Activation('softmax')
        self.start_voc_id = start_voc_id
        self.input_length = input_length
        self.output_length = output_length
        self.recurrent_units = recurrent_units

    def shiftLeftSentence(
        self,
        sentence: ndarray,
        ) -> ndarray:
        '''shift target sequence to learn'''
        shape = tf.shape(sentence)
        batchs = shape[0]
        zero_pad = tf.expand_dims(tf.repeat([0],repeats=[batchs]), 1)
        seq = sentence[:,1:]
        result = tf.concat([seq,zero_pad],1)
        return result

    def call(
        self,
        inputs,
        training=None,
        mask=None,
        ):
        '''forward step'''
        enc_inputs,dec_inputs = inputs
        #print('--------forward step---------')
        #print(inputs)
        #if isinstance(inputs, tuple) or isinstance(inputs, list):
        #    print(train_data)
        #    raise Exception('error')
        enc_outputs,states = self.encoder(enc_inputs,training)
        #print('enc_outputs',enc_outputs.shape)
        #dec_inputs = self.shiftSentence(trues)

        #print('enc_out_states from encoder:',states[0].shape)
        outputs, _, _ = self.decoder(dec_inputs,training,
            initial_state=states,enc_outputs=enc_outputs)
        #
        #outputs = self.out(outputs)
        return outputs

    def train_step(
        self,
        train_data: tuple
    ) -> dict:
        '''train step callback'''
        inputs,trues = train_data
        #dec_inputs = self.shiftSentence(trues)
        dec_inputs = trues
        sft_trues = self.shiftLeftSentence(trues)

        with tf.GradientTape() as tape:
            #print('====================-')
            #print(trues)
            outputs = self((inputs,dec_inputs),training=True)
            loss = self.compiled_loss(
                sft_trues,outputs,
                regularization_losses=self.losses)

        variables = self.trainable_variables

        gradients = tape.gradient(loss, variables)

        self.optimizer.apply_gradients(zip(gradients, variables))

        self.compiled_metrics.update_state(sft_trues, outputs)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """
        The logic for one evaluation step.
        """
        inputs,trues = data
        #dec_inputs = self.shiftSentence(y)
        dec_inputs = trues
        sft_trues = self.shiftLeftSentence(trues)

        outputs = self((inputs,dec_inputs), training=False)
        # Updates stateful loss metrics.
        self.compiled_loss(
            sft_trues, outputs, regularization_losses=self.losses)

        self.compiled_metrics.update_state(sft_trues, outputs)
        return {m.name: m.result() for m in self.metrics}

#    def translate(
#        self,
#        sentence: ndarray) -> ndarray:
#        '''translate sequence'''
#        input_length = sentence.size
#        sentence = sentence.reshape([1,input_length])
#        enc_outputs,states=self.encoder(sentence,training=True)
#        dec_input = tf.expand_dims([self.start_voc_id], 0)
#        target_sentence =[]
#        attention_plot = np.zeros([self.output_length, input_length])
#        for t in range(self.output_length):
#            predictions, states, rnn_sequence = self.decoder(dec_input,
#                training=False,initial_state=states,enc_outputs=enc_outputs)
#            voc_id = tf.argmax(predictions[0,0]).numpy()
#            target_sentence.append(voc_id)
#            # recalc attention weight
#            #print('states',states[0].shape)
#            #print('enc_outputs',enc_outputs.shape)
#            score = tf.matmul(rnn_sequence,enc_outputs,transpose_b=True)
#            attention_plot[t,:] = tf.nn.softmax(score[0,0,:]).numpy()
#            if targ_lang.index_word[voc_id] == '<end>':
#                break
#            dec_input = tf.expand_dims([voc_id], 0)
#
#        return (np.array(target_sentence), attention_plot)
#
#
    def translate(self, sentence, dataset):
        result, sentence, attention_plot = self.evaluate_sentence(sentence, dataset)

        print('Input: %s' % (sentence))
        print('Predicted translation: {}'.format(result))

        attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
        self.plot_attention(attention_plot, sentence.split(' '), result.split(' '))

    def evaluate_sentence(self,sentence, dataset):
        attention_plot = np.zeros((self.output_length, self.input_length))

        sentence = dataset.preprocess_sentence(sentence)

        inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=self.input_length,
                                                         padding='post')
        inputs = tf.convert_to_tensor(inputs)

        result = ''

        hidden = [tf.zeros((1, self.recurrent_units))]
        enc_out, enc_hidden = self.encoder(inputs, training=False, initial_state=hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.start_voc_id], 0)

        for t in range(self.output_length):
            predictions, dec_hidden, rnn_sequence = self.decoder(
                dec_input, training=False, initial_state=dec_hidden,enc_outputs=enc_out)

            attention_weights = tf.matmul(rnn_sequence,enc_out,transpose_b=True)
            #print('attention_weights',attention_weights.shape)
            attention_weights = tf.nn.softmax(attention_weights[0,0,:])

            # storing the attention weights to plot later on
            attention_weights = tf.reshape(attention_weights, (-1, ))
            attention_plot[t] = attention_weights.numpy()

            predicted_id = tf.argmax(predictions[0,0]).numpy()

            if result!='':
                result += ' '
            result += targ_lang.index_word[predicted_id]

            if targ_lang.index_word[predicted_id] == '<end>':
                return result, sentence, attention_plot

            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims([predicted_id], 0)

        return result, sentence, attention_plot

    # function for plotting the attention weights
    def plot_attention(self, attention, sentence, predicted_sentence):
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1, 1, 1)

        attention = attention[:len(predicted_sentence), :len(sentence)]
        ax.matshow(attention, cmap='viridis')

        fontdict = {'fontsize': 14}

        ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
        ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.show()


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)


num_examples=5000#30000
epochs = 10#10
batch_size = 64
word_vect_size=128#256
recurrent_units=256#1024

dataset = EngFraDataset()

print("Generating data...")
input_tensor, target_tensor, inp_lang, targ_lang = dataset.load_data(num_examples=num_examples)
input_vocab_size = max(len(inp_lang.index_word)+1,)
#print ("Input Language; index to word mapping")
#dataset.convert(inp_lang, input_tensor[0])
#print ()
#print ("Target Language; index to word mapping")
#dataset.convert(targ_lang, target_tensor[0])
#print ()


corpus_size = len(input_tensor)
print("embedding_dim:",word_vect_size)
print("units:",recurrent_units)
print("Total questions:", corpus_size)
#input_voc,target_voc,input_dic,target_dic=dataset.dicts()
print("Input word dictionary:",len(inp_lang.index_word)+1)
print("Target word dictionary:",len(targ_lang.index_word)+1)
# Explicitly set apart 10% for validation data that we never train over.
#split_at = len(questions) - len(questions) // 10
#(x_train, x_val) = questions[:split_at], questions[split_at:]
#(y_train, y_val) = answers[:split_at], answers[split_at:]

input_length  = input_tensor.shape[1] #DIGITS*2 + 1;
output_length = target_tensor.shape[1] #DIGITS + 1;
print("Input length:",input_length)
print("Output length:",output_length)


encoder = Encoder(input_length,len(inp_lang.word_index)+1, word_vect_size, recurrent_units)
# sample input
sample_input_batch = input_tensor[0:batch_size]
sample_dec_input_batch = target_tensor[0:batch_size]
sample_hidden = encoder.initialize_hidden_state(batch_size)
sample_output, sample_hidden = encoder(sample_input_batch, sample_hidden)

decoder = Decoder(output_length,len(targ_lang.word_index)+1, word_vect_size, recurrent_units)
sample_decoder_output, _, _ = decoder(sample_dec_input_batch,True,
                                      initial_state=sample_hidden, enc_outputs=sample_output)
print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

#exit()

seq2seq = Seq2seq(
    input_length=input_length,
    input_vocab_size=len(inp_lang.word_index)+1,
    output_length=output_length,
    target_vocab_size=len(targ_lang.word_index)+1,
    start_voc_id=targ_lang.word_index['<start>'],
    word_vect_size=word_vect_size,
    recurrent_units=recurrent_units,
)


print("Compile model...")
seq2seq.compile(
    #loss='sparse_categorical_crossentropy',
    #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    loss=loss_function,
    optimizer='adam',
    metrics=['accuracy'],
    )

checkpoint_filepath = './seq2seq-attention-translation/ckpt'

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='auto',
    save_best_only=True)

#if os.path.exists(checkpoint_filepath):
#    print("Loading weights")
#    model.load_weights(checkpoint_filepath)

print("Train model...")
history = seq2seq.fit(
    input_tensor,
    target_tensor,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1,
    #callbacks=[checkpoint],
)

#for i in range(10):
#    idx = np.random.randint(0,len(input_tensor))
#    question = input_tensor[idx]
#    input = question.reshape(1,input_length)
#
#    #input = keras.utils.to_categorical(
#    #    input.reshape(input.size,),
#    #    num_classes=len(input_voc)
#    #    ).reshape(input.shape[0],input.shape[1],len(input_voc))
#
#    #predict = model.predict(input)
#    #predict_seq = np.argmax(predict[0].reshape(output_length,len(target_dic)),axis=1)
#    predict_seq,attention = seq2seq.translate(question);
#
#    predict_str = dataset.seq2str(predict_seq,targ_lang)
#    question_str = dataset.seq2str(question,inp_lang)
#    answer_str = dataset.seq2str(target_tensor[idx],targ_lang)
#    #correct = '*' if predict_str==answer_str else ' '
#    #print('%s=%s : %s %s' % (question_str,predict_str,correct,answer_str))
#    print('Input    : {}'.format(question_str))
#    print('Predicted: {}'.format(predict_str))
#    print('Correct  : {}'.format(answer_str))
#    print()
#    seq2seq.plot_attention(attention, question_str.split(' '), predict_str.split(' '))

for i in range(10):
    idx = np.random.randint(0,len(input_tensor))
    question = input_tensor[idx]
    #input = question.reshape(1,max_length_inp)
    #input = keras.utils.to_categorical(
    #    input.reshape(input.size,),
    #    num_classes=len(input_voc)
    #    ).reshape(input.shape[0],input.shape[1],len(input_voc))

    #predict = model.predict(input)
    #predict_seq = np.argmax(predict[0].reshape(output_length,len(target_dic)),axis=1)
    sentence = dataset.seq2str(question,inp_lang)
    predict_seq = seq2seq.translate(sentence, dataset);
    answer = target_tensor[idx]
    sentence = dataset.seq2str(answer,targ_lang)
    print('Target: %s' % (sentence))
    print()


plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['accuracy'],label='accuracy')
plt.plot(history.history['val_loss'],label='val_loss')
plt.plot(history.history['val_accuracy'],label='val_accuracy')
plt.legend()
plt.show()
