<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;

$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);
$plt = new Plot(null,$mo);

#$loss = $nn->losses()->CategoricalCrossEntropy();
$loss = $nn->losses()->BinaryCrossEntropy();
$p = $mo->array([
    [0.02],[0.001],
]);
$t = $mo->array([
    0.0,0.0,
]);
echo $loss->loss($t,$p)."\n";
