<?php
namespace RindowTest\NeuralNetworks\Activation\SigmoidTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Activation\Sigmoid;


class Test extends TestCase
{
    public function testNormal()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $activation = new Sigmoid($backend);

        $x = $mo->array([-1.0,-0.5,0.0,0.5,1.0]);
        $y = $activation->forward($x, $training=true);

        $this->assertEquals([-1.0,-0.5,0.0,0.5,1.0],$x->toArray());
        $this->assertTrue($y[0]<0.5);
        $this->assertTrue($y[1]<0.5);
        $this->assertTrue($y[2]==0.5);
        $this->assertTrue($y[3]>0.5);
        $this->assertTrue($y[4]>0.5);

        $dout = $x;
        $dx = $activation->backward($dout);
        $this->assertTrue(abs(-0.196-$dx[0])<0.01);
        $this->assertTrue(abs(-0.117-$dx[1])<0.01);
        $this->assertTrue($dx[2]==0.0);
        $this->assertTrue(abs(0.117-$dx[3])<0.01);
        $this->assertTrue(abs(0.196-$dx[4])<0.01);
    }
}
