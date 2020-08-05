<?php
namespace RindowTest\NeuralNetworks\Activation\ReLUTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Activation\ReLU;

class Test extends TestCase
{
    public function testNormal()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $activation = new ReLU($backend);

        $inputs = $mo->array([
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
        ]);
        $copyInputs = $mo->copy($inputs);
        $outputs = $activation->call($inputs, $training=true);
        $this->assertEquals([4,5],$outputs->shape());
        $this->assertEquals(
            [0.0,0.0,0.0,0.5,1.0],
            $outputs[0]->toArray()
        );
        $this->assertEquals($copyInputs->toArray(),$inputs->toArray());


        $dOutputs = $mo->array([
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
        ]);
        $copydOutputs = $mo->copy($dOutputs);
        $dInputs = $activation->differentiate($dOutputs);
        $this->assertEquals([4,5],$dInputs->shape());
        $this->assertEquals(
            [0.0,0.0,0.0,0.5,1.0],
            $dInputs[0]->toArray()
        );
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());

    }
}
