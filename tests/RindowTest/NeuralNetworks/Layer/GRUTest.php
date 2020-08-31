<?php
namespace RindowTest\NeuralNetworks\Layer\GRUTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\GRU;
use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Activation\Tanh;

class Test extends TestCase
{
    public function testDefaultInitialize()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $layer = new GRU(
            $backend,
            $units=4,
            [
                'input_shape'=>[5,3],
            ]);

        $layer->build();
        $params = $layer->getParams();
        $this->assertCount(3,$params);
        $this->assertEquals([3,4*3],$params[0]->shape());
        $this->assertEquals([4*3,4],$params[1]->shape());
        $this->assertEquals([4*3],$params[2]->shape());

        $grads = $layer->getGrads();
        $this->assertCount(3,$grads);
        $this->assertEquals([3,4*3],$grads[0]->shape());
        $this->assertEquals([4*3,4],$grads[1]->shape());
        $this->assertEquals([4*3],$grads[2]->shape());
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
        $layer = new GRU(
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
        $layer = new GRU(
            $backend,
            $units=4,
            [
            ]);
        $layer->build($inputShape=[5,3]);

        //$this->assertEquals([3],$layer->inputShape());
        $this->assertEquals([4],$layer->outputShape());
    }

    public function testInitializeWithReturnSequence()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $layer = new GRU(
            $backend,
            $units=4,
            [
                'input_shape'=>[5,3],
                'return_sequences'=>true,
                'return_state'=>true,
            ]);
        $layer->build();

        //$this->assertEquals([3],$layer->inputShape());
        $this->assertEquals([5,4],$layer->outputShape());
    }

    public function testDefaultForwardAndBackword()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $fn = $backend;

        $layer = new GRU(
            $backend,
            $units=4,
            [
                'input_shape'=>[5,3],
            ]);

        $layer->build();
        $grads = $layer->getGrads();
        
        
        //
        // forward
        //
        //  2 batch
        $inputs = $mo->ones([6,5,3]);
        $initialStates = [$mo->ones([6,4])];
        $copyInputs = $mo->copy($inputs);
        $copyStates = [
            $mo->copy($initialStates[0])];
        $outputs = $layer->forward($inputs,$training=true, $initialStates
        );
        // 
        $this->assertEquals([6,4],$outputs->shape());
        $this->assertEquals($copyInputs->toArray(),$inputs->toArray());

        //
        // backword
        //
        // 2 batch
        $dOutputs =
            $mo->ones([6,4]);
        $dStates =
            [$mo->ones([6,4])];

        $copydOutputs = $mo->copy(
            $dOutputs);
        $copydStates = [
            $mo->copy($dStates[0])];
        $dInputs = $layer->backward($dOutputs,$dStates);
        // 2 batch
        $this->assertEquals([6,5,3],$dInputs->shape());
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

    public function testForwardAndBackwordWithReturnSeqquence()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $fn = $backend;

        $layer = new GRU(
            $backend,
            $units=4,
            [
                'input_shape'=>[5,3],
                'return_sequences'=>true,
                'return_state'=>true,
            ]);

        $layer->build();
        $grads = $layer->getGrads();
        
        
        //
        // forward
        //
        //  2 batch
        $inputs = $mo->ones([6,5,3]);
        $initialStates = [
            $mo->ones([6,4])];
        $copyInputs = $mo->copy($inputs);
        $copyStates = [
            $mo->copy($initialStates[0])];
        [$outputs,$nextStates] = $layer->forward($inputs,$training=true, $initialStates
        );
        // 
        $this->assertEquals([6,5,4],$outputs->shape());
        $this->assertCount(1,$nextStates);
        $this->assertEquals([6,4],$nextStates[0]->shape());
        $this->assertEquals($copyInputs->toArray(),$inputs->toArray());
        $this->assertEquals($copyStates[0]->toArray(),$initialStates[0]->toArray());

        //
        // backword
        //
        // 2 batch
        $dOutputs =
            $mo->ones([6,5,4]);
        $dStates = [
            $mo->ones([6,4])];

        $copydOutputs = $mo->copy(
            $dOutputs);
        $copydStates = [
            $mo->copy($dStates[0])];
        [$dInputs,$dPrevStates] = $layer->backward($dOutputs,$dStates);
        // 2 batch
        $this->assertEquals([6,5,3],$dInputs->shape());
        $this->assertCount(1,$dPrevStates);
        $this->assertEquals([6,4],$dPrevStates[0]->shape());
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

    public function testForwardAndBackwordWithReturnSeqquenceWithoutInitialStates()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $fn = $backend;

        $layer = new GRU(
            $backend,
            $units=4,
            [
                'input_shape'=>[5,3],
                'return_sequences'=>true,
                'return_state'=>true,
            ]);

        $layer->build();
        $grads = $layer->getGrads();
        
        
        //
        // forward
        //
        //  2 batch
        $inputs = $mo->ones([6,5,3]);
        $initialStates = null;
        $copyInputs = $mo->copy($inputs);
        //$copyStates = [$mo->copy($initialStates[0])];
        [$outputs,$nextStates] = $layer->forward($inputs,$training=true, $initialStates
        );
        // 
        $this->assertEquals([6,5,4],$outputs->shape());
        $this->assertCount(1,$nextStates);
        $this->assertEquals([6,4],$nextStates[0]->shape());
        $this->assertEquals($copyInputs->toArray(),$inputs->toArray());
        //$this->assertEquals($copyStates[0]->toArray(),$initialStates[0]->toArray());

        //
        // backword
        //
        // 2 batch
        $dOutputs =
            $mo->ones([6,5,4]);
        $dStates =
            [$mo->ones([6,4])];

        $copydOutputs = $mo->copy(
            $dOutputs);
        $copydStates = [
            $mo->copy($dStates[0])];
        [$dInputs,$dPrevStates] = $layer->backward($dOutputs,$dStates);
        // 2 batch
        $this->assertEquals([6,5,3],$dInputs->shape());
        $this->assertCount(1,$dPrevStates);
        $this->assertEquals([6,4],$dPrevStates[0]->shape());
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

        $layer = new GRU(
            $backend,
            $units=4,
            [
                'input_shape'=>[3,5],
                'return_sequences'=>true,
                'return_state'=>true,
                'activation'=>null,
                'recurrent_activation'=>null,
            ]);

        $kernel = $mo->ones([5,4*3]);
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
        $inputs = $mo->ones([2,3,5]);
        $states = [
            $mo->ones([2,4])];
        [$outputs,$nextStates] = $layer->forward($inputs,$training=true, $states);
        // 
        $this->assertEquals(
            [2,3,4],
            $outputs->shape());
        $this->assertEquals(
            [2,4],
            $nextStates[0]->shape());
        //
        // backword
        //
        // 2 batch
        $dOutputs =
            $mo->ones([2,3,4]);
        $dStates =
            [$mo->ones([2,4])];

        [$dInputs,$dPrevStates] = $layer->backward($dOutputs,$dStates);
        // 2 batch
        $this->assertEquals(
            [2,3,5],
            $dInputs->shape());
        $this->assertEquals(
            [2,4],
            $dPrevStates[0]->shape());
        $this->assertEquals(
            [5,16],
            $grads[0]->shape());
        $this->assertEquals(
            [4,16],
            $grads[1]->shape());
        $this->assertEquals(
            [16],
            $grads[2]->shape());
    }
}