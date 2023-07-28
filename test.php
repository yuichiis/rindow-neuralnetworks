<?php
include __DIR__.'/samples/neural-machine-translation-with-transformer.php';

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\Math\Plot\Plot;

$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);
$plt = new Plot([],$mo);
$K = $nn->backend();
$g = $nn->gradient();

//$a = $mo->array(1);
//echo $mo->toString($a)."\n";

//$batchSize = 5;
//$wordVectSize = 16;
//$depth = 4;
//$num_heads = 2;
//$inputLen = 3;
//$targetLen = 7;
//$mha = new MultiHeadAttention($K,$nn,$depth,$num_heads);
//
//$target = $g->Variable($K->ones([$batchSize,$targetLen,$wordVectSize]));
//$source = $g->Variable($K->ones([$batchSize,$inputLen,$wordVectSize]));
//$mask = $mo->ones([$batchSize,$inputLen],NDArray::int8);
//
//[$output,$weights] = $mha($source,$source,$target,mask:$mask);

//echo implode(',',$output->shape())."\n";
//echo implode(',',$weights->shape())."\n";

//$shape = [2,2,2];
//$a = $mo->arange((int)array_product($shape),dtype:NDArray::int32)->reshape($shape);
//$la = $mo->la();
//echo implode(',',$a->shape())."\n";
////echo $mo->toString($a,null,true)."\n";
//$a = $la->repeat($a,3,axis:-1);
//echo $mo->toString($a,null,true)."\n";
//echo implode(',',$a->shape())."\n";
//$enclayer = new EncoderLayer($K,$nn,
//
//);

//$x = $mo->array(
//    [[[1,2],
//      [0,0]],
//     [[0,0],
//      [0,0]]]
//);
//echo '==== input ===='."\n";
//echo $mo->toString($x,'%6.6f',true).'('.implode(',',$x->shape()).')'."\n";
//echo '==== layer norm rindow ===='."\n";
//$shape = $x->shape();
//$size = array_pop($shape);
//$mean = $K->mean($x,axis:-1);
//$mean = $K->expandDims($mean,axis:-1);
//echo $mo->toString($mean,'%6.6f',true).'('.implode(',',$mean->shape()).')'."\n";
//echo '===='."\n";
//$mean = $K->repeat($mean,$size,axis:-1);
//$mean = $K->squeeze($mean,axis:-1);
//echo $mo->toString($mean,'%6.6f',true).'('.implode(',',$mean->shape()).')'."\n";
//echo '===='."\n";
//$var = $K->mean($K->square($K->sub($x,$mean)),axis:-1);
//$var = $K->squeeze($K->repeat($K->expandDims($var,axis:-1),$size,axis:-1),axis:-1);
//echo '===='."\n";
//$norm = $K->div($K->sub($x,$mean),$K->sqrt($K->increment($var,1e-7)));
//echo $mo->toString($norm,'%6.6f',true).'('.implode(',',$norm->shape()).')'."\n";

//$bnorm = $nn->layers->BatchNormalization();
//$x = $mo->array(
//    [[[1,0],
//      [2,0]],
//     [[3,0],
//      [4,0]]]
//);
//echo '==== input ===='."\n";
//echo $mo->toString($x,'%6.6f',true).'('.implode(',',$x->shape()).')'."\n";
//echo '===='."\n";
//$norm = $bnorm($x,training:true);
//echo $mo->toString($norm,'%6.6f',true).'('.implode(',',$norm->shape()).')'."\n";
//########################################################
//$bnorm = $nn->layers->LayerNormalization();
//$x = $mo->array(
//    [[[1,2],
//      [3,4]],
//     [[0,0],
//      [0,0]]]
//);
//echo '==== input ===='."\n";
//echo $mo->toString($x,'%6.6f',true).'('.implode(',',$x->shape()).')'."\n";
//echo '===='."\n";
//$norm = $bnorm($x,training:true);
//echo $mo->toString($norm,'%6.6f',true).'('.implode(',',$norm->shape()).')'."\n";

//$vocab_size = 8;
//$wordVectSize = 4;
//$emb = new PositionalEmbedding(
//  $K,
//  $nn,
//  $vocab_size,
//  $wordVectSize,
//);
//
//$inputs = $K->array([
//  [1,2,3,4, 5,0,0,0],
//  [1,2,3,4, 5,6,7,0],
//],dtype:NDArray::int8);
//
//$vects = $emb($inputs);


//$enclayer = new EncoderLayer($K,$nn,
//  $wordVectSize=4, 
//  $num_heads=2,
//  $dff=8, // units
//  $dropout_rate=0.1);
//
//$batchSize = 2;
//$inputLen  = 8;
//$x = $mo->ones([$batchSize,$inputLen,$wordVectSize]);
//$mask = $mo->ones([$batchSize,$inputLen],NDArray::int8);
//$y = $enclayer($x,$training=true,$mask);
//

//$enc = new Encoder(
//  $K,
//  $nn,
//  $numLayers=2,
//  $wordVectSize=4,
//  $num_heads=2,
//  $dff=4,
//  $vocabSize=8,
//  $maximumPositionEncoding=null,
//  $inputLength=8,
//  $dropout_rate=0.1,
//);
//
//$inputs = $K->array([
//  [1,2,3,4, 5,0,0,0],
//  [1,2,3,4, 5,6,7,0],
//],dtype:NDArray::int8);
//$mask = $K->array([
//  [true,true,true,true, true,false,false,false],
//  [true,true,true,true, true,true, true, false],
//],dtype:NDArray::bool);
//
//$outputs = $enc(
//  $inputs,
//  $training=true,
//  $mask
//);

//$a = $enc->positionalEncoding($maxLength=7, $depth=8);
//echo $mo->toString($a,'%5.5e',true)."\n";

//use Rindow\NeuralNetworks\Gradient\Core\ArrayShape;

//$shape = new ArrayShape([1,2,3]);
//
//foreach($shape as $i => $v) {
//  echo "$i=>$v\n";
//}
//var_dump($shape[1]);
//var_dump(count($shape));

//$batchSize = 2;
//$targetLen = 8;
//$dec = new DecoderLayer(
//  $K,
//  $nn,
//  $wordVectSize=4,
//  $num_heads=2,
//  $dff=4,
//  $dropout_rate=0.1,
//);
//
//$x = $K->ones([$batchSize, $targetLen, $wordVectSize]);
//$enc_output = $K->ones([$batchSize, $targetLen, $wordVectSize]);
//$look_ahead_mask = $K->array([
//  [true,true,true,true, true,false,false,false],
//  [true,true,true,true, true,true, true, false],
//],dtype:NDArray::bool);
//$padding_mask = $K->array([
//  [true,true,true,true, true,false,false,false],
//  [true,true,true,true, true,true, true, false],
//],dtype:NDArray::bool);
//
//$dec(
//  $x,                 # (batch_size, target_seq_len, wordVectSize)
//  $enc_output,        # (batch_size, target_seq_len, wordVectSize)
//  $training=true,
//  $look_ahead_mask,   # (batch_size, target_seq_len)
//  $padding_mask,      # (batch_size, target_seq_len)
//);

//$batchSize = 2;
//$targetLen = 8;
//$numLayers = 2;
//$wordVectSize = 4;
//$num_heads = 2;
//$dff = 4;
//$target_vocab_size = 8;
//
//$dec = new Decoder(
//  $K,
//  $nn,
//  $numLayers,
//  $wordVectSize,
//  $num_heads,
//  $dff,
//  $target_vocab_size,
//  inputLength:8,
//);
//
//$inputs = $K->ones([$batchSize, $targetLen]);
//$enc_output = $K->ones([$batchSize, $targetLen, $wordVectSize]);
//$look_ahead_mask = $K->array([
//  [[[true,true,true,true, true,false,false,false]]],
//  [[[true,true,true,true, true,true, true, false]]],
//],dtype:NDArray::bool); // (batchSize, 1, 1, targetLen)
//$padding_mask = $K->array([
//  [[[true,true,true,true, true,false,false,false]]],
//  [[[true,true,true,true, true,true, true, false]]],
//],dtype:NDArray::bool); // (batchSize, 1, 1, targetLen)
//
//$dec(
//  $inputs,
//  $enc_output,
//  $training=true,
//  $look_ahead_mask,
//  $padding_mask,
//);

//$la = $mo->la();
//$a = $la->ones($la->alloc([3,3]));
//$b = $la->ones($la->alloc([3,1]));
//$la->trmm($a,$b);
//echo $mo->toString($a,'%3.3f',true)."\n";
//echo $mo->toString($b,'%3.3f',true)."\n";




//$la = $mo->la();
//$shape = [2,2,2];
//$a = $mo->arange((int)array_product($shape),dtype:NDArray::int32);
//echo $mo->toString($a,'%2d',true)."\n";
//$a = $a->reshape($shape);
//echo "inputs=[".implode(',',$a->shape())."]\n";
//echo $mo->toString($a,'%2d',true)."\n";
////$a = $la->repeat($a,3,axis:2);
//$a = $la->repeat($a,3,axis:2,keepdims:true);
//echo "outputs=[".implode(',',$a->shape())."]\n";
//echo $mo->toString($a,'%2d',true)."\n";

//$shape = [2,2,2];
//$repeats = 3;
//$axis = 2;
////$outShape = [2,2,6];
//$outerShape = array_slice($shape,0,$axis);
////$base = $shape[$axis];
////$base /= $repeats;
//$innerShape = array_slice($shape,$axis);
//$inputs = array_merge($outerShape,[$repeats],$innerShape);
//echo 'shape:'.implode(',',$shape)."\n";
//echo 'outerShape:'.implode(',',$outerShape)."\n";
//echo 'repeats:'.$repeats."\n";
////echo 'base:'.$base."\n";
//echo 'innerShape:'.implode(',',$innerShape)."\n";
//echo 'inputs:'.implode(',',$inputs)."\n";

//$opencl = new Rindow\Math\Matrix\Drivers\OpenCLExt\OpenCLFactory();
//$info = new Rindow\Math\Matrix\CLInfo($opencl);
//$info->info();
//exit();

//$batchSize = 2;
//$inputLength = 8;
//$targetLen = 8;
//$numLayers = 2;
//$wordVectSize = 4;
//$num_heads = 2;
//$dff = 4;
//$input_vocab_size = 8;
//$target_vocab_size = 8;
//
//$transformer = new Transformer(
//    $K,
//    $nn,
//    $numLayers,
//    $wordVectSize,
//    $num_heads,
//    $dff,
//    $input_vocab_size,
//    $target_vocab_size,
//    $inputLength,
//    $targetLen,
//);
//
//$inputs = $g->Variable($mo->array([
//    [1,2,3,0,0,0,0,0],
//    [1,2,3,4,0,0,0,0],
//]));
//$targets = $g->Variable($mo->array([
//    [6,7,8,9,0,0,0,0],
//    [6,7,8,0,0,0,0,0],
//]));

//$transformer($inputs,$targets,true);


//$decaySteps = 1;
//$decayRate = 0.001;
//$lr = 0.001;
//$init_lr = 0.002;
//$rms = [];
//$itd = [];
//$x = [];
//for($step=0;$step<1000;$step++) {
//    $x[] = $step;
//    $itd[] = $init_lr / (1 + $decayRate * ($step / $decaySteps));
//    $rms[] = $lr * (1 / (1 + $decayRate * $step));
//}
//$x = $mo->array($x);
//
//$plt->plot($x,$mo->array($rms),label:'rms');
//$plt->plot($x,$mo->array($itd),label:'itd');
//$plt->legend();
//$plt->show();

//$lossfunc = $nn->losses->SparseCategoricalCrossEntropy();
////$lossfunc = $nn->losses->SparseCategoricalCrossEntropy(reduction:'none');
//$predicts = $K->array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]]);
//$trues = $K->array([1, 2],dtype:NDArray::int32);
//
//$loss = $lossfunc($trues, $predicts);
//echo $mo->toString($loss)."\n";

//$la = $mo->la();
//$x = $mo->array([
//    [1,2,3],
//    [4,5,6],
//],dtype:NDArray::int32);
//
//$y = make_labels($la,$x);
//
//echo "==x==\n";
//echo $mo->toString($x).$mo->dtypeToString($x->dtype())."\n";
//echo "==y==\n";
//echo $mo->toString($y).$mo->dtypeToString($y->dtype())."\n";


//===================================================
$calcac = new CustomAccuracy($nn);

$pred = $K->array([
    [[0,1,0,0],[0,0,1,0],[1,0,0,0]],
    [[0,0,1,0],[0,0,0,1],[1,0,0,0]],
]);
$label = $K->array([
    [1,2,0],
    [2,0,0],
],dtype:NDArray::int32);

$ac = $calcac($label,$pred);
echo "label=".$K->toString($label)."\n";
echo "accuracy=".$ac."\n";

$label = $K->array([
    [1,2,0],
    [2,1,0],
],dtype:NDArray::int32);

$ac = $calcac($label,$pred);
echo "label=".$K->toString($label)."\n";
echo "accuracy=".$ac."\n";
