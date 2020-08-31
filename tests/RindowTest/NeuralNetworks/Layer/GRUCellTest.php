<?php
namespace RindowTest\NeuralNetworks\Layer\GRUCellTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\GRUCell;
use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Activation\Tanh;

class Test extends TestCase
{
    public function testDefaultInitialize()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $layer = new GRUCell(
            $backend,
            $units=4,
            [
                'input_shape'=>[3]
            ]);

        $layer->build();
        $params = $layer->getParams();
        $this->assertCount(3,$params);
        $this->assertEquals([3,12],$params[0]->shape());
        $this->assertEquals([12,4],$params[1]->shape());
        $this->assertEquals([12],$params[2]->shape());

        $grads = $layer->getGrads();
        $this->assertCount(3,$grads);
        $this->assertEquals([3,12],$grads[0]->shape());
        $this->assertEquals([12,4],$grads[1]->shape());
        $this->assertEquals([12],$grads[2]->shape());
        $this->assertNull(
            $layer->getActivation()
            );

        //$this->assertEquals([3],$layer->inputShape());
        $this->assertEquals([4],$layer->outputShape());
    }

    public function testNotspecifiedInputShape()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $layer = new GRUCell(
            $backend,
            $units=4,
            [
            ]);

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Input shape is not defined');
        $layer->build();
    }

    public function testSetInputShape()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $layer = new GRUCell(
            $backend,
            $units=4,
            [
            ]);
        $layer->build($inputShape=[3]);

        //$this->assertEquals([3],$layer->inputShape());
        $this->assertEquals([4],$layer->outputShape());
    }

    public function testNormalForwardAndBackword()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $fn = $backend;

        $layer = new GRUCell(
            $backend,
            $units=4,
            [
                'input_shape'=>[3]
            ]);

        $layer->build();
        $grads = $layer->getGrads();
        
        
        //
        // forward
        //
        //  2 batch
        $inputs = $mo->ones([2,3]);
        $states = [$mo->ones([2,4])];
        $object = new \stdClass();
        $copyInputs = $mo->copy($inputs);
        $copyStates = [
            $mo->copy($states[0])];
        [$outputs,$nextStates] = $layer->forward($inputs, $states,$training=true,$object);
        // 
        $this->assertEquals([2,4],$outputs->shape());
        $this->assertCount(1,$nextStates);
        $this->assertEquals([2,4],$nextStates[0]->shape());
        $this->assertEquals($copyInputs->toArray(),$inputs->toArray());
        $this->assertEquals($copyStates[0]->toArray(),$states[0]->toArray());

        //
        // backword
        //
        // 2 batch
        $dOutputs =
            $mo->ones([2,4]);
        $dStates =
            [$mo->ones([2,4])];

        $copydOutputs = $mo->copy(
            $dOutputs);
        $copydStates = [
            $mo->copy($dStates[0])];
        [$dInputs,$dPrevStates] = $layer->backward($dOutputs,$dStates,$object);
        // 2 batch
        $this->assertEquals([2,3],$dInputs->shape());
        $this->assertCount(1,$dPrevStates);
        $this->assertEquals([2,4],$dPrevStates[0]->shape());
        $this->assertNotEquals(
            $mo->zerosLike($grads[0])->toArray(),
            $grads[0]->toArray());
        $this->assertNotEquals(
            $mo->zerosLike($grads[1])->toArray(),
            $grads[1]->toArray());
        $this->assertNotEquals(
            $mo->zerosLike($grads[2])->toArray(),
            $grads[2]->toArray());
        
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
        $this->assertEquals($copydStates[0]->toArray(),$dStates[0]->toArray());
    }

    public function testOutputsAndGrads()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $fn = $backend;

        $layer = new GRUCell(
            $backend,
            $units=4,
            [
                'input_shape'=>[3],
                'activation'=>null,
                'recurrent_activation'=>null,
            ]);

        $kernel = $mo->ones([3,4*3]);
        $recurrent = $mo->ones([4*3,4]);
        $bias = $mo->ones([4*3]);
        $layer->build(null,
            ['sampleWeights'=>[$kernel,$recurrent,$bias]]
        );
        $this->assertNull($layer->getActivation());
        $grads = $layer->getGrads();
        
        
        //
        // forward
        //
        //  2 batch
        $inputs = $mo->ones([2,3]);
        $states = [$mo->ones([2,4])];
        $object = new \stdClass();
        [$outputs,$nextStates] = $layer->forward($inputs, $states,$training=true,$object);
        // 
        $this->assertEquals([
            [576,576,576,576],
            [576,576,576,576],
            ],$outputs->toArray());
        $this->assertEquals([
            [576,576,576,576],
            [576,576,576,576],
            ],$nextStates[0]->toArray());
        //
        // backword
        //
        // 2 batch
        $dOutputs =
            $mo->ones([2,4]);
        $dStates =
            [$mo->ones([2,4])];

        [$dInputs,$dPrevStates] = $layer->backward($dOutputs,$dStates,$object);
        // 2 batch
        $this->assertEquals([
            [1732,1732,1732],
            [1732,1732,1732],
            ],$dInputs->toArray());
        $this->assertEquals([
            [1732,1732,1732,1732],
            [1732,1732,1732,1732],
            ],$dPrevStates[0]->toArray());
        $this->assertEquals([
            [272,272,272,272,
             272,272,272,272,
             288,288,288,288],
            [272,272,272,272,
             272,272,272,272,
             288,288,288,288],
            [272,272,272,272,
             272,272,272,272,
             288,288,288,288],
            ],$grads[0]->toArray());
        $this->assertEquals([
            [272,272,272,272,
             272,272,272,272,
             288,288,288,288],
            [272,272,272,272,
             272,272,272,272,
             288,288,288,288],
            [272,272,272,272,
             272,272,272,272,
             288,288,288,288],
            [272,272,272,272,
             272,272,272,272,
             288,288,288,288],
            ],$grads[1]->toArray());
        $this->assertEquals(
            [272,272,272,272,
             34,34,34,34,
             272,272,272,272,
             288,288,288,288],
            $grads[2]->toArray());
    }
}
