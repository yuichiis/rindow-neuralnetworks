<?php
include __DIR__ . '/vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);

// 入力は [1, 5, 5, 1] です。
//$input = $mo->array(array_fill(0, 5, array_fill(0, 5, 0)));
//$input = $input->reshape([1, 5, 5, 1]);
$input = $mo->zeros([1,6,6,4]);

echo "input shape:".$mo->shapeToString($input->shape()) . PHP_EOL;
//echo $mo->toString($input,indent:true) . PHP_EOL;

$conv = $nn->layers()->Conv2D(
    filters: 2,
    kernel_size: 3,
    strides: 1,
    padding: 'same',
    //input_shape: [5, 5, 1]
    input_shape: [6, 6, 4]
);

// この行でエラーが発生します: 出力形状が一致しません: (5,5,1)、(5,5,1,1) である必要があります
$output = $conv->forward($input);
echo "output shape:".$mo->shapeToString($output->shape()) . PHP_EOL;

// 質問:同じパディング計算中に、サイズが1の場合にバッチ次元が圧縮されるのは既知の動作ですか？

// Conv2Dをpadding: 'same'と組み合わせて使用​​する際に、
// 出力が(バッチ次元を含めて)4Dランクを厳密に維持するようにするための、
// 標準的な動作例を提供していただけますか？

echo "input shape:".$mo->shapeToString($input->shape()) . PHP_EOL;

$pooling = $nn->layers()->MaxPooling2D(
    pool_size: 2,
    strides: 1,
    padding: 'same',
    //input_shape: [5, 5, 1]
    input_shape: [6, 6, 4]
);

$output = $pooling->forward($input);
echo "output shape:".$mo->shapeToString($output->shape()) . PHP_EOL;
