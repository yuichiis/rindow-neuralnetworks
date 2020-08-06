<?php
namespace RindowTest\NeuralNetworks\Loss\BinaryCrossEntropyTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Loss\BinaryCrossEntropy;
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
            'Rindow\NeuralNetworks\Loss\BinaryCrossEntropy',
            $nn->losses()->BinaryCrossEntropy());
    }

    public function testDefault()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $func = new BinaryCrossEntropy($backend);

        $x = $mo->array([
            [0.0], [0.0] , [10.0],
        ]);
        $t = $mo->array([
            0.0, 0.0 , 1.0,
        ]);
        $y = $backend->sigmoid($x);
        $loss = $func->loss($t,$y);
        $accuracy = $func->accuracy($t,$x);
        $this->assertLessThan(0.001,abs(0.0-$loss));

        $dx = $backend->dSigmoid($func->differentiateLoss(),$y);
        $this->assertLessThan(0.0001,1-$mo->asum($mo->op($mo->op($y->reshape([3]),'-',$dx->reshape([3])),'-',$t)));


        $x = $mo->array([
            [0.0], [0.0] , [10.0],
        ]);
        $t = $mo->array([
            0.0, 1.0 , 0.0,
        ]);
        $y = $backend->sigmoid($x);
        $loss = $func->loss($t,$y);
        $this->assertLessThan(0.01,abs(0.23-$loss));

        $dx = $backend->dSigmoid($func->differentiateLoss(),$y);
        $this->assertLessThan(0.0001,1-$mo->asum($mo->op($mo->op($y->reshape([3]),'-',$dx->reshape([3])),'-',$t)));
    }

    public function testFromLogits()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $func = new BinaryCrossEntropy($backend);
        $func->setFromLogits(true);

        $x = $mo->array([
            [0.0], [0.0] , [10.0],
        ]);
        $t = $mo->array([
            0.0, 0.0 , 1.0,
        ]);
        $y = $func->call($x,true);
        $loss = $func->loss($t,$y);
        $this->assertLessThan(0.01,abs(0.0-$loss));

        $dx = $func->differentiateLoss();
        $this->assertLessThan(0.0001,$mo->asum($mo->op($mo->op($y->reshape([3]),'-',$dx->reshape([3])),'-',$t)));

        $x = $mo->array([
            [0.0], [0.0] , [10.0],
        ]);
        $t = $mo->array([
            0.0, 1.0 , 0.0,
        ]);
        $y = $func->call($x,true);
        $loss = $func->loss($t,$y);
        $this->assertLessThan(0.01,abs(0.23-$loss));

        $dx = $func->differentiateLoss();
        $this->assertLessThan(0.0001,$mo->asum($mo->op($mo->op($y->reshape([3]),'-',$dx->reshape([3])),'-',$t)));
    }
}
