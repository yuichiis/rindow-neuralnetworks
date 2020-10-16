<?php
namespace RindowTest\NeuralNetworks\Layer\AttentionTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\Attention;
use InvalidArgumentException;

class Test extends TestCase
{
    public function testDefaultInitialize()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $layer = new Attention(
            $backend,
            [
                'input_shapes'=>[[3,2],[4,2]],
            ]);

        $layer->build();
        $params = $layer->getParams();
        $this->assertCount(0,$params);

        $grads = $layer->getGrads();
        $this->assertCount(0,$grads);

        $this->assertEquals([3,2],$layer->outputShape());
    }

    public function testInitializeWithReturnAttentionScores()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $layer = new Attention(
            $backend,
            [
                'input_shapes'=>[[3,2],[4,2]],
                'return_attention_scores'=>true,
            ]);

        $shapes = $layer->build();
        $params = $layer->getParams();
        $this->assertCount(0,$params);

        $grads = $layer->getGrads();
        $this->assertCount(0,$grads);

        $this->assertEquals([3,2],$layer->outputShape());
        $this->assertEquals([3,4],$shapes[1]);
    }

    public function testNotspecifiedInputShape()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $layer = new Attention(
            $backend,
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
        $layer = new Attention(
            $backend,
            [
            ]);
        // [batch,3,2],[batch,4,2]
        $layer->build($inputShape=[[3,2],[4,2]]);
        // [batch,3,4]
        $this->assertEquals([3,2],$layer->outputShape());
    }

    public function testNormalForwardAndBackward()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $fn = $backend;

        $layer = new Attention(
            $backend,
            [
                'return_attention_scores'=>true,
            ]);

        $layer->build($inputShape=[[2,3],[4,3]]);

        //
        // forward
        //
        //  batch size 2
        $query = $mo->array([
            [[1,0,0],[0,1,0]],
            [[1,0,0],[0,1,0]],
        ]);
        $value = $mo->array([
            [[1,0,0],[0,1,0],[0,0,1],[0,0,0]],
            [[1,0,0],[0,1,0],[0,0,1],[0,0,0]],
        ]);
        $inputs = [$query,$value];
        $copyInputs = [$mo->copy($query),$mo->copy($value)];
        [$outputs,$scores] = $layer->forward($inputs, $training=true);
        //
        $this->assertEquals([2,2,4],$scores->shape());
        $this->assertEquals([2,2,3],$outputs->shape());
        $this->assertEquals($copyInputs[0]->toArray(),$inputs[0]->toArray());
        $this->assertEquals($copyInputs[1]->toArray(),$inputs[1]->toArray());
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->softmax($mo->array([
                [1,0,0,0],[0,1,0,0],
                [1,0,0,0],[0,1,0,0],
            ])),
            $scores->reshape([4,4])));
        $this->assertTrue($mo->la()->isclose(
            $mo->array([
                [[0.475367,0.174878,0.174878],[0.174878,0.475367,0.174878]],
                [[0.475367,0.174878,0.174878],[0.174878,0.475367,0.174878]],
            ]),
            $outputs));
        //
        // backward
        //
        // 2 batch
        $dOutputs = $mo->ones($outputs->shape());

        $copydOutputs = $mo->copy(
            $dOutputs);
        $dInputs = $layer->backward($dOutputs);
        // 2 batch
        $this->assertCount(2,$dInputs);
        $this->assertEquals([2,2,3],$dInputs[0]->shape());
        $this->assertEquals([2,4,3],$dInputs[1]->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
        $this->assertTrue($mo->la()->isclose(
            $mo->array([
                [[0.08313105, 0.0305822 , 0.0305822],
                 [0.0305822 , 0.08313105, 0.0305822]],
                [[0.08313105, 0.0305822 , 0.0305822],
                 [0.0305822 , 0.08313105, 0.0305822]],
            ]),
            $dInputs[0]));
        $this->assertTrue($mo->la()->isclose(
            $mo->array([
                [[0.73337567, 0.68082684, 0.65024465],
                 [0.68082684, 0.73337567, 0.65024465],
                 [0.38033763, 0.38033763, 0.34975544],
                 [0.20545992, 0.20545992, 0.34975544]],
                [[0.73337567, 0.68082684, 0.65024465],
                 [0.68082684, 0.73337567, 0.65024465],
                 [0.38033763, 0.38033763, 0.34975544],
                 [0.20545992, 0.20545992, 0.34975544]],
            ]),
            $dInputs[1]));
    }
}
