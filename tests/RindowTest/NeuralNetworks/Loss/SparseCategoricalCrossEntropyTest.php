<?php
namespace RindowTest\NeuralNetworks\Loss\SparseCategoricalCrossEntropyTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Loss\SparseCategoricalCrossEntropy;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;

class Test extends TestCase
{
    public function testBuilder()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $this->assertInstanceof(
            'Rindow\NeuralNetworks\Loss\SparseCategoricalCrossEntropy',
            $nn->losses()->SparseCategoricalCrossEntropy());
    }

    public function testDefault()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $func = new SparseCategoricalCrossEntropy($backend);

        $x = $mo->array([
            [0.0, 0.0 , 6.0],
            [0.0, 0.0 , 6.0],
        ]);
        $t = $mo->array([2, 2],NDArray::int64);
        $y = $backend->softmax($x);
        $loss = $func->loss($t,$y);
        $accuracy = $func->accuracy($t,$x);

        $this->assertLessThan(0.01,abs(0.0-$loss));

        $dx = $backend->dsoftmax($func->differentiateLoss(),$y);
        $this->assertLessThan(0.0001,$mo->asum($mo->op($mo->op($y,'-',$dx),'-',
            $backend->onehot($t,$x->shape()[1]))));

        $x = $mo->array([
            [0.0, 0.0 , 6.0],
            [0.0, 0.0 , 6.0],
        ]);
        $t = $mo->array([1, 1]);
        $y = $backend->softmax($x);
        $loss = $func->loss($t,$y);
        $this->assertLessThan(0.01,abs(6.0-$loss));

        $dx = $backend->dsoftmax($func->differentiateLoss(),$y);
        $this->assertLessThan(0.001,$mo->asum($mo->op($mo->op($y,'-',$dx),'-',
            $backend->onehot($t,$x->shape()[1]))));
    }

    public function testFromLogits()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $func = new SparseCategoricalCrossEntropy($backend);
        $func->setFromLogits(true);

        $x = $mo->array([
            [0.0, 0.0 , 6.0],
            [0.0, 0.0 , 6.0],
        ]);
        $t = $mo->array([2, 2],NDArray::int64);
        $y = $func->call($x,true);
        $loss = $func->loss($t,$y);
        $accuracy = $func->accuracy($t,$x);

        $this->assertLessThan(0.01,abs(0.0-$loss));

        $dx = $func->differentiateLoss();
        $this->assertLessThan(0.0001,$mo->asum($mo->op($mo->op($y,'-',$dx),'-',
            $backend->onehot($t,$x->shape()[1]))));

        $x = $mo->array([
            [0.0, 0.0 , 6.0],
            [0.0, 0.0 , 6.0],
        ]);
        $t = $mo->array([1, 1]);
        $y = $func->call($x,true);
        $loss = $func->loss($t,$y);
        $this->assertLessThan(0.01,abs(6.0-$loss));

        $dx = $func->differentiateLoss();
        $this->assertLessThan(0.0001,$mo->asum($mo->op($mo->op($y,'-',$dx),'-',
            $backend->onehot($t,$x->shape()[1]))));
    }

}
