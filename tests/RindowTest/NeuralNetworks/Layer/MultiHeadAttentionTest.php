<?php
namespace RindowTest\NeuralNetworks\Layer\MultiHeadAttentionTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\MultiHeadAttention;
use InvalidArgumentException;
use WeakMap;

class MultiHeadAttentionTest extends TestCase
{
    public function newMatrixOperator()
    {
        return new MatrixOperator();
    }

    public function newNeuralNetworks($mo)
    {
        return new NeuralNetworks($mo);
    }

    public static function providerDefaultInitialize()
    {
        return [
            "input_key" => [[
                "num_heads"=>2,
                "key_dim"=>5,
                "value_dim"=>null,
                "use_bias"=>null,
                "dropout"=>null,
                "query_shape"=>[2, 8, 16],
                "value_shape"=>[2, 4, 16],
                "expected_output_shape"=>[2, 8, 16],
                "expected_num_trainable_weights"=>8,
                "expected_num_non_trainable_weights"=>0,
                "expected_num_seed_generators"=>0,
                "expected_num_losses"=>0,
                "supports_masking"=>true,
                "run_training_check"=>false,
            ]],
            "input_key_and_value" => [[
                "num_heads"=>2,
                "key_dim"=>5,
                "value_dim"=>6,
                "use_bias"=>false,
                "dropout"=>0.5,
                "query_shape"=>[2, 8, 16],
                "value_shape"=>[2, 4, 16],
                "expected_output_shape"=>[2, 8, 16],
                "expected_num_trainable_weights"=>4,
                "expected_num_non_trainable_weights"=>0,
                "expected_num_seed_generators"=>0,
                "expected_num_losses"=>0,
                "supports_masking"=>true,
                "run_training_check"=>false,
            ]],
        ];
    }

    /**
    * @dataProvider providerDefaultInitialize
    */
    public function testDefaultInitialize($params)
    {
        extract($params);
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();

        $batch_size = array_shift($query_shape);
        array_shift($value_shape);

        $layer = new MultiHeadAttention(
            $K,
            $num_heads, // num_heads
            $key_dim,   // key_dim
            value_dim:$value_dim,
            use_bias:$use_bias,
            input_shapes:[
                $query_shape, // query_shape
                $value_shape, // value_shape
            ],
        );
        $inputs = [
            $g->Variable($K->zeros(array_merge([$batch_size],$query_shape))),
            $g->Variable($K->zeros(array_merge([$batch_size],$value_shape))),
        ];

        $layer->build($inputs);
        $params = $layer->getParams();
        $this->assertCount($expected_num_trainable_weights,$params);
        //$this->assertEquals($expected_num_non_trainable_weights,$params[0]->shape());

        $grads = $layer->getGrads();
        $this->assertCount($expected_num_trainable_weights,$grads);

        array_shift($expected_output_shape);
        $this->assertEquals($expected_output_shape,$layer->outputShape());
    }

    /**
    * @dataProvider providerDefaultInitialize
    */
    public function testSetInputShape($params)
    {
        extract($params);
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $batch_size = array_shift($query_shape);
        array_shift($value_shape);
        $layer = new MultiHeadAttention(
            $K,
            $num_heads, // num_heads
            $key_dim,   // key_dim
            value_dim:$value_dim,
            use_bias:$use_bias,
        );
        $inputs = [
            $g->Variable($K->zeros(array_merge([$batch_size],$query_shape))),
            $g->Variable($K->zeros(array_merge([$batch_size],$value_shape))),
        ];
        $layer->build($inputs);
        array_shift($expected_output_shape);
        $this->assertEquals($expected_output_shape,$layer->outputShape());
    }

    public function testUnmatchSpecifiedInputShape()
    {
        $num_heads = 2;
        $key_dim = 5;
        $query_shape = [2, 8, 16];
        $value_shape = [2, 4, 16];

        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $batch_size = array_shift($query_shape);
        array_shift($value_shape);
        $layer = new MultiHeadAttention(
            $K,
            $num_heads, // num_heads
            $key_dim,   // key_dim
            input_shapes:[
                [8, 32], // query_shape
                [8, 16], // value_shape
            ],
        );
        $inputs = [
            $g->Variable($K->zeros(array_merge([$batch_size],$query_shape))),
            $g->Variable($K->zeros(array_merge([$batch_size],$value_shape))),
        ];
    
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Input shape is inconsistent: defined as ((8,32),(8,16)) but ((8,16),(4,16)) given in MultiHeadAttention');
        $layer->build($inputs);
    }

    public function testNormalForwardAndBackward()
    {
        $num_heads = 8;
        $key_dim = 4;
        $full_query_shape = [2, 6, 16];
        $full_value_shape = [2, 7, 16];

        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $la = $K->primaryLA();
        $layer = new MultiHeadAttention(
            $K,
            $num_heads, // num_heads
            $key_dim,   // key_dim
            //kernel_initializer:'ones',
            //bias_initializer:'zeros',
        );
        $query = $g->Variable($la->randomNormal($full_query_shape,mean:0,scale:1));
        $value = $g->Variable($la->randomNormal($full_value_shape,mean:0,scale:1));
        //$query = $g->Variable($K->ones($full_query_shape));
        //$value = $g->Variable($K->ones($full_value_shape));
        //$query = $g->Variable($K->increment(
        //        $K->scale(0.5, $K->array($mo->la()->range(array_product($full_query_shape),dtype:NDArray::float32)
        //        ->reshape($full_query_shape))),
        //    1,
        //));
        //$value = $g->Variable($K->increment(
        //        $K->scale(0.2, $K->array($mo->la()->range(array_product($full_value_shape),dtype:NDArray::float32)
        //        ->reshape($full_value_shape))),
        //    1,
        //));
        $inputs = [
            $query,
            $value,
        ];
        [$batches,$tSeq,$dim] = $full_query_shape;
        [$batches,$sSeq,$dim] = $full_value_shape;

        $layer->build($inputs,
            //sampleWeights:[
            //    $K->ones([$num_heads,  $key_dim, $dim]),    // query_dense/kernel
            //    $K->zeros([$num_heads, $key_dim]),          // query_dense/bias
            //    $K->ones([$num_heads,  $key_dim, $dim]),    // key_dense/kernel
            //    $K->zeros([$num_heads, $key_dim]),          // key_dense/bias
            //    $K->ones([$num_heads,  $key_dim, $dim]),    // value_dense/kernel
            //    $K->zeros([$num_heads, $key_dim]),          // value_dense/bias
            //    $K->ones([$num_heads,  $key_dim, $dim]),    // output_dense/kernel
            //    $K->zeros([$dim]),                          // output_dense/bias
            //]
        );

        //
        // forward
        //
        //  batch size 2
        //echo "query: ".$mo->toString($query,indent:true)."\n";
        //////////////echo "key: ".$mo->toString($key,indent:true)."\n";
        //echo "value: ".$mo->toString($value,indent:true)."\n";


        $salt = $mo->la()->range(array_product($full_query_shape),dtype:NDArray::float32)
                ->reshape($full_query_shape);
        $salt = $g->Variable($salt);
        $copyInputs = [$K->copy($query),$K->copy($value)];
        [$outputsVariable,$scores,$resultValiable] = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$layer,$inputs,$salt) {
                [$outputsVariable,$scores] = $layer->forward(
                    $inputs,
                    training:true,
                    returnAttentionScores:true,
                );
                $resultValiable = $g->mul($outputsVariable,$salt);
                return [$outputsVariable,$scores,$resultValiable];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //echo 'outputs:'.$mo->toString($outputs,format:'%10.7f',indent:true)."\n";
        //echo 'scores:'.$mo->toString($scores,indent:true)."\n";
        //echo 'kernel:'.$mo->toString($layer->getParams()[0],format:'%14.7f',indent:true)."\n";
        $this->assertEquals([$batches, $num_heads, $tSeq, $sSeq],$scores->shape());
        $this->assertEquals([$batches, $tSeq, $dim],$outputs->shape());
        $this->assertEquals($copyInputs[0]->toArray(),$inputs[0]->toArray());
        $this->assertEquals($copyInputs[1]->toArray(),$inputs[1]->toArray());
        //$this->assertTrue($mo->la()->isclose(
        //    $K->fill([2,8,6,7], 0.14285715),
        //    $K->ndarray($scores)));
        //$this->assertTrue($mo->la()->isclose(
        //    //$K->mul($salt,$K->fill($full_query_shape,512)),
        //    $K->fill($full_query_shape,512),
        //    $K->ndarray($outputs)
        //));
        //
        // backward
        //
        // 2 batch
        $dResultValiable = $K->ones($resultValiable->shape());
        [$dOutputs,$dSalt] = $resultValiable->creator()->backward([$dResultValiable]);
        $this->assertEquals($outputsVariable->shape(),$dOutputs->shape());
        $this->assertEquals($resultValiable->shape(),$dSalt->shape());
        $this->assertTrue($mo->la()->isclose(
            $dOutputs,$salt
        ));
        $this->assertTrue($mo->la()->isclose(
            $dSalt,$outputs
        ));

        $copydOutputs = $K->copy($dOutputs);
        $dInputs = $outputsVariable->creator()->backward([$dOutputs]);
        // 2 batch
        $this->assertCount(2,$dInputs);
        //echo "dQuery: ".$mo->shapeToString($dInputs[0]->shape())."\n";
        //echo "dValue: ".$mo->shapeToString($dInputs[1]->shape())."\n";
        $this->assertEquals($full_query_shape,$dInputs[0]->shape());
        $this->assertEquals($full_value_shape,$dInputs[1]->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());

        //echo "dQuery: ".$mo->toString($dInputs[0],indent:true)."\n";
    }

    public function testCausalMask()
    {
        $num_heads = 8;
        $key_dim = 4;
        $full_query_shape = [2, 6, 16];
        $full_value_shape = [2, 7, 16];

        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new MultiHeadAttention(
            $K,
            $num_heads, // num_heads
            $key_dim,   // key_dim
            kernel_initializer:'ones',
            bias_initializer:'zeros',
        );
        $inputs = [
            $g->Variable($K->zeros($full_query_shape)),
            $g->Variable($K->zeros($full_value_shape)),
        ];
        [$batches,$tSeq,$dim] = $full_query_shape;
        [$batches,$sSeq,$dim] = $full_value_shape;

        $layer->build($inputs,
            //sampleWeights:[
            //    $K->ones([$num_heads,  $key_dim, $dim]),    // query_dense/kernel
            //    $K->zeros([$num_heads, $key_dim]),          // query_dense/bias
            //    $K->ones([$num_heads,  $key_dim, $dim]),    // key_dense/kernel
            //    $K->zeros([$num_heads, $key_dim]),          // key_dense/bias
            //    $K->ones([$num_heads,  $key_dim, $dim]),    // value_dense/kernel
            //    $K->zeros([$num_heads, $key_dim]),          // value_dense/bias
            //    $K->ones([$num_heads,  $key_dim, $dim]),    // output_dense/kernel
            //    $K->zeros([$dim]),                          // output_dense/bias
            //]
        );

        //
        // forward
        //
        //  batch size 2
        $query = $g->Variable($K->ones($full_query_shape));
        $value = $g->Variable($K->ones($full_value_shape));
        $inputs = [$query,$value];
        $copyInputs = [$K->copy($query),$K->copy($value)];
        [$outputsVariable,$scores] = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs) {
                [$outputsVariable,$scores] = $layer->forward(
                    $inputs,
                    training:true,
                    returnAttentionScores:true,
                    useCausalMask:true,
                );
                return [$outputsVariable,$scores];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //
        $this->assertEquals([$batches, $num_heads, $tSeq, $sSeq],$scores->shape());
        $this->assertEquals([$batches, $tSeq, $dim],$outputs->shape());
        $this->assertEquals($copyInputs[0]->toArray(),$inputs[0]->toArray());
        $this->assertEquals($copyInputs[1]->toArray(),$inputs[1]->toArray());
        //echo $mo->shapeToString($scores->shape())."\n";
        //echo $mo->toString($K->ndarray($scores),format:'%8.4f',indent:true)."\n";
        //echo $mo->shapeToString($outputs->shape())."\n";
        //echo $mo->toString($K->ndarray($outputs),format:'%8.1f',indent:true)."\n";

        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($outputs),
            $K->scale(512,$K->ones($full_query_shape))
        ));

        $sc = $mo->array([
                [  1.00000000,  0.00000000,  0.00000000,  0.00000000,  0.00000000,  0.00000000,  0.0000],
                [  0.50000000,  0.50000000,  0.00000000,  0.00000000,  0.00000000,  0.00000000,  0.0000],
                [  0.33333334,  0.33333334,  0.33333334,  0.00000000,  0.00000000,  0.00000000,  0.0000],
                [  0.25000000,  0.25000000,  0.25000000,  0.25000000,  0.00000000,  0.00000000,  0.0000],
                [  0.20000000,  0.20000000,  0.20000000,  0.20000000,  0.20000000,  0.00000000,  0.0000],
                [  0.16666667,  0.16666667,  0.16666667,  0.16666667,  0.16666667,  0.16666667,  0.0000]
        ]);
        //echo $mo->shapeToString($sc->shape())."\n";
        $sc = $mo->la()->repeat($sc,$num_heads,axis:0);
        $sc = $mo->la()->repeat($sc,$batches,axis:0);
        //echo $mo->toString($sc,format:'%8.4f',indent:true)."\n";
        //echo $mo->shapeToString($sc->shape())."\n";
        //echo $mo->shapeToString($scores->shape())."\n";
        $this->assertEquals($sc->shape(),$scores->shape());
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($scores),
            $sc,
        ));


        //
        // backward
        //
        // 2 batch
        $dOutputs = $K->ones($outputs->shape());

        $copydOutputs = $K->copy(
            $dOutputs);
        $dInputs = $outputsVariable->creator()->backward([$dOutputs]);
        // 2 batch
        $this->assertCount(2,$dInputs);
        //echo "dQuery: ".$mo->shapeToString($dInputs[0]->shape())."\n";
        //echo "dValue: ".$mo->shapeToString($dInputs[1]->shape())."\n";
        $this->assertEquals($full_query_shape,$dInputs[0]->shape());
        $this->assertEquals($full_value_shape,$dInputs[1]->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
    }

    public function testMaskBoth()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Attention($K);
        $inputs = [
            $g->Variable($K->zeros([2,3,4])),
            $g->Variable($K->zeros([2,5,4])),
        ];

        $layer->build($inputs);

        //
        // forward
        //
        //  batch size 2
        $query = $K->ones([2,3,4]);
        $value = $K->ones([2,5,4]);
        $queryMask = $K->array([ // (2,3)
            [true,true, false],
            [true,false,false],
        ],dtype:NDArray::bool);
        $valueMask = $K->array([ // (2,5)
            [true,true,false,false,false],
            [true,true,true, true, false],
        ],dtype:NDArray::bool);
        $inputs = [$query,$value];
        $copyInputs = [$K->copy($query),$K->copy($value)];
        [$outputsVariable,$scores] = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs,$queryMask,$valueMask) {
                [$outputsVariable,$scores] = $layer->forward($inputs, mask:[$queryMask,$valueMask],
                                returnAttentionScores:true);
                return [$outputsVariable,$scores];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //
        $this->assertEquals([2,3,5],$scores->shape());
        $this->assertEquals([2,3,4],$outputs->shape());
        $this->assertEquals($copyInputs[0]->toArray(),$inputs[0]->toArray());
        $this->assertEquals($copyInputs[1]->toArray(),$inputs[1]->toArray());
        $this->assertTrue($mo->la()->isclose(
            $mo->array(
                [[[0.5, 0.5, 0.0, 0.0, 0.0],
                  [0.5, 0.5, 0.0, 0.0, 0.0],
                  [0.5, 0.5, 0.0, 0.0, 0.0]],
                  [[0.25, 0.25, 0.25, 0.25, 0.0  ],
                   [0.25, 0.25, 0.25, 0.25, 0.0  ],
                   [0.25, 0.25, 0.25, 0.25, 0.0  ]]]
            ),
            $scores = $K->ndarray($scores)
        ));
        $this->assertTrue($mo->la()->isclose(
            $mo->array(
                [[[1.0, 1.0, 1.0, 1.0],
                  [1.0, 1.0, 1.0, 1.0],
                  [0.0, 0.0, 0.0, 0.0]],
                 [[1.0, 1.0, 1.0, 1.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]]]
            ),
            $outputs = $K->ndarray($outputs)
        ));
        //
        // backward
        //
        // 2 batch
        $dOutputs = $K->ones($outputs->shape());

        $copydOutputs = $K->copy(
            $dOutputs);
        $dInputs = $outputsVariable->creator()->backward([$dOutputs]);
        // 2 batch
        $this->assertCount(2,$dInputs);
        $this->assertEquals([2,3,4],$dInputs[0]->shape());
        $this->assertEquals([2,5,4],$dInputs[1]->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
        $this->assertTrue($mo->la()->isclose(
            $mo->array(
                [[[0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]],
                 [[0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]]]
            ),
            $K->ndarray($dInputs[0])));
        $this->assertTrue($mo->la()->isclose(
            $mo->array(
                [[[1.0,   1.0,   1.0,   1.0 ],
                  [1.0,   1.0,   1.0,   1.0 ],
                  [0.0,   0.0,   0.0,   0.0 ],
                  [0.0,   0.0,   0.0,   0.0 ],
                  [0.0,   0.0,   0.0,   0.0 ]],
                 [[0.25,  0.25,  0.25,  0.25],
                  [0.25,  0.25,  0.25,  0.25],
                  [0.25,  0.25,  0.25,  0.25],
                  [0.25,  0.25,  0.25,  0.25],
                  [0.0,   0.0,   0.0,   0.0 ]]]
            ),
            $K->ndarray($dInputs[1])));
    }

    public function testMaskFloatBoth()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Attention($K);
        $inputs = [
            $g->Variable($K->zeros([2,3,4])),
            $g->Variable($K->zeros([2,5,4])),
        ];

        $layer->build($inputs);

        //
        // forward
        //
        //  batch size 2
        $query = $K->ones([2,3,4]);
        $value = $K->ones([2,5,4]);
        $queryMask = $K->array([ // (2,3)
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]);
        $valueMask = $K->array([ // (2,5)
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0],
        ]);
        $inputs = [$query,$value];
        $copyInputs = [$K->copy($query),$K->copy($value)];
        [$outputsVariable,$scores] = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs,$queryMask,$valueMask) {
                [$outputsVariable,$scores] = $layer->forward($inputs, mask:[$queryMask,$valueMask],
                                returnAttentionScores:true);
                return [$outputsVariable,$scores];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //
        $this->assertEquals([2,3,5],$scores->shape());
        $this->assertEquals([2,3,4],$outputs->shape());
        $this->assertEquals($copyInputs[0]->toArray(),$inputs[0]->toArray());
        $this->assertEquals($copyInputs[1]->toArray(),$inputs[1]->toArray());
        $this->assertTrue($mo->la()->isclose(
            $mo->array(
                [[[0.5, 0.5, 0.0, 0.0, 0.0],
                  [0.5, 0.5, 0.0, 0.0, 0.0],
                  [0.5, 0.5, 0.0, 0.0, 0.0]],
                  [[0.25, 0.25, 0.25, 0.25, 0.0  ],
                   [0.25, 0.25, 0.25, 0.25, 0.0  ],
                   [0.25, 0.25, 0.25, 0.25, 0.0  ]]]
            ),
            $scores = $K->ndarray($scores)
        ));
        $this->assertTrue($mo->la()->isclose(
            $mo->array(
                [[[1.0, 1.0, 1.0, 1.0],
                  [1.0, 1.0, 1.0, 1.0],
                  [0.0, 0.0, 0.0, 0.0]],
                 [[1.0, 1.0, 1.0, 1.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]]]
            ),
            $outputs = $K->ndarray($outputs)
        ));
        //
        // backward
        //
        // 2 batch
        $dOutputs = $K->ones($outputs->shape());

        $copydOutputs = $K->copy(
            $dOutputs);
        $dInputs = $outputsVariable->creator()->backward([$dOutputs]);
        // 2 batch
        $this->assertCount(2,$dInputs);
        $this->assertEquals([2,3,4],$dInputs[0]->shape());
        $this->assertEquals([2,5,4],$dInputs[1]->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
        $this->assertTrue($mo->la()->isclose(
            $mo->array(
                [[[0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]],
                 [[0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]]]
            ),
            $K->ndarray($dInputs[0])));
        $this->assertTrue($mo->la()->isclose(
            $mo->array(
                [[[1.0,   1.0,   1.0,   1.0 ],
                  [1.0,   1.0,   1.0,   1.0 ],
                  [0.0,   0.0,   0.0,   0.0 ],
                  [0.0,   0.0,   0.0,   0.0 ],
                  [0.0,   0.0,   0.0,   0.0 ]],
                 [[0.25,  0.25,  0.25,  0.25],
                  [0.25,  0.25,  0.25,  0.25],
                  [0.25,  0.25,  0.25,  0.25],
                  [0.25,  0.25,  0.25,  0.25],
                  [0.0,   0.0,   0.0,   0.0 ]]]
            ),
            $K->ndarray($dInputs[1])));
    }

    public function testMaskDoNotExpandMask()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Attention($K,do_not_expand_mask:true);

        $query = $K->ones([2,2,3,2]);
        $value = $K->ones([2,2,5,2]);

        $inputs = [
            $g->Variable($query),
            $g->Variable($value),
        ];
        $layer->build($inputs);

        //
        // forward
        //
        //  batch size 2
        $queryMask = $K->array([ // (2,1,3,1)
            [[[true],[true], [false]]],
            [[[true],[false],[false]]],
        ]);
        $valueMask = $K->array([ // (2,1,1,5)
            [[[true,true,false,false,false]]],
            [[[true,true,true, true, false]]],
        ]);
        $inputs = [$query,$value];
        $copyInputs = [$K->copy($query),$K->copy($value)];
        [$outputsVariable,$scores] = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs,$queryMask,$valueMask) {
                [$outputsVariable,$scores] = $layer->forward($inputs, mask:[$queryMask,$valueMask],
                                returnAttentionScores:true);
                return [$outputsVariable,$scores];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //
        $this->assertEquals([2,2,3,5],$scores->shape());
        $this->assertEquals([2,2,3,2],$outputs->shape());
        $this->assertEquals($copyInputs[0]->toArray(),$inputs[0]->toArray());
        $this->assertEquals($copyInputs[1]->toArray(),$inputs[1]->toArray());
        //
        // backward
        //
        // 2 batch
        $dOutputs = $K->ones($outputs->shape());

        $copydOutputs = $K->copy(
            $dOutputs);
        $dInputs = $outputsVariable->creator()->backward([$dOutputs]);
        // 2 batch
        $this->assertCount(2,$dInputs);
        $this->assertEquals([2,2,3,2],$dInputs[0]->shape());
        $this->assertEquals([2,2,5,2],$dInputs[1]->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
    }

    public function testMaskOneSide()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Attention($K);
        $inputs = [
            $g->Variable($K->zeros([2,2,3])),
            $g->Variable($K->zeros([2,4,3])),
        ];

        $layer->build($inputs);

        $query = $K->array([
            [[1,0,0],[0,1,0]],
            [[1,0,0],[0,1,0]],
        ]);
        $value = $K->array([
            [[1,0,0],[0,1,0],[0,0,1],[0,0,0]],
            [[1,0,0],[0,1,0],[0,0,1],[0,0,0]],
        ]);
        $queryMask = $K->array([
            [true,false],
            [false,true],
        ]);
        $valueMask = $K->array([
            [false,false,true,true],
            [false,true,true,false],
        ]);
        $inputs = [$query,$value];
        $copyInputs = [$K->copy($query),$K->copy($value)];

        //
        //  queryMask
        //
        // forward
        //
        [$outputsVariable,$scores] = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs,$queryMask,$valueMask) {
                [$outputsVariable,$scores] = $layer->forward($inputs, mask:[$queryMask,null],
                                returnAttentionScores:true);
                return [$outputsVariable,$scores];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //
        $this->assertEquals([2,2,4],$scores->shape());
        $this->assertEquals([2,2,3],$outputs->shape());
        $this->assertEquals($copyInputs[0]->toArray(),$inputs[0]->toArray());
        $this->assertEquals($copyInputs[1]->toArray(),$inputs[1]->toArray());
        //
        // backward
        //
        $dOutputs = $K->ones($outputs->shape());

        $copydOutputs = $K->copy(
            $dOutputs);
        $dInputs = $outputsVariable->creator()->backward([$dOutputs]);
        // 2 batch
        $this->assertCount(2,$dInputs);
        $this->assertEquals([2,2,3],$dInputs[0]->shape());
        $this->assertEquals([2,4,3],$dInputs[1]->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());


        //
        //  valueMask
        //
        // forward
        //
        [$outputsVariable,$scores] = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs,$queryMask,$valueMask) {
                [$outputsVariable,$scores] = $layer->forward($inputs, mask:[null,$valueMask],
                                returnAttentionScores:true);
                return [$outputsVariable,$scores];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //
        $this->assertEquals([2,2,4],$scores->shape());
        $this->assertEquals([2,2,3],$outputs->shape());
        $this->assertEquals($copyInputs[0]->toArray(),$inputs[0]->toArray());
        $this->assertEquals($copyInputs[1]->toArray(),$inputs[1]->toArray());
        //
        // backward
        //
        $dOutputs = $K->ones($outputs->shape());

        $copydOutputs = $K->copy(
            $dOutputs);
        $dInputs = $outputsVariable->creator()->backward([$dOutputs]);
        // 2 batch
        $this->assertCount(2,$dInputs);
        $this->assertEquals([2,2,3],$dInputs[0]->shape());
        $this->assertEquals([2,4,3],$dInputs[1]->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
    }

    public function testUseScale()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Attention($K,use_scale:true);
        $inputs = [
            $g->Variable($K->zeros([2,2,3])),
            $g->Variable($K->zeros([2,4,3])),
        ];
        $layer->build($inputs);

        //
        // forward
        //
        $query = $K->array(
            [[[1,0,0],[0,1,0]],
             [[1,0,0],[0,1,0]]]
        );
        $value = $K->array(
            [[[1,0,0],[0,1,0],[0,0,1],[0,0,0]],
             [[1,0,0],[0,1,0],[0,0,1],[0,0,0]]]
        );
        $inputs = [$query,$value];
        $copyInputs = [$K->copy($query),$K->copy($value)];
        [$outputsVariable,$scoresVariable] = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs) {
                [$outputsVariable,$scoresVariable] = $layer->forward($inputs,
                                returnAttentionScores:true);
                return [$outputsVariable,$scoresVariable];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        $scores = $K->ndarray($scoresVariable);

        //
        $this->assertEquals([2,2,4],$scores->shape());
        $this->assertEquals([2,2,3],$outputs->shape());
        $this->assertEquals($copyInputs[0]->toArray(),$inputs[0]->toArray());
        $this->assertEquals($copyInputs[1]->toArray(),$inputs[1]->toArray());
        $this->assertTrue($mo->la()->isclose(
            $mo->array(
                [[[0.47536692, 0.17487772, 0.17487772, 0.17487772],
                  [0.17487772, 0.47536692, 0.17487772, 0.17487772]],
                 [[0.47536692, 0.17487772, 0.17487772, 0.17487772],
                  [0.17487772, 0.47536692, 0.17487772, 0.17487772]]]
            ),
            $K->ndarray($scores)
        ));
        $this->assertTrue($mo->la()->isclose(
            $mo->array(
                [[[0.47536692, 0.17487772, 0.17487772],
                  [0.17487772, 0.47536692, 0.17487772]],
                 [[0.47536692, 0.17487772, 0.17487772],
                  [0.17487772, 0.47536692, 0.17487772]]]
            ),
            $K->ndarray($outputs)
        ));

        //
        // backward
        //
        $dOutputs = [
            $K->ones($outputs->shape()),
            $K->ones($scores->shape()),
        ];

        $variables = $layer->trainableVariables();
        $grads = new WeakMap();
        $copydOutputs = [];
        $copydOutputs[] = $K->copy($dOutputs[0]);
        $copydOutputs[] = $K->copy($dOutputs[1]);
        $dInputs = $outputsVariable->creator()->backward($dOutputs,$grads,$variables);
        // 2 batch
        $this->assertCount(2,$dInputs);
        $this->assertEquals([2,2,3],$dInputs[0]->shape());
        $this->assertEquals([2,4,3],$dInputs[1]->shape());
        $this->assertEquals($copydOutputs[0]->toArray(),$dOutputs[0]->toArray());
        $this->assertEquals($copydOutputs[1]->toArray(),$dOutputs[1]->toArray());

        $this->assertCount(1,$variables);
        $this->assertCount(1,$grads);
        $this->assertEquals(1.0, $K->scalar($variables[0]->value()));
        $this->assertLessThan(1e-6,(0.33252418-$K->scalar($grads[$variables[0]])));
    }

    public function testMultiHead()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Attention($K,use_scale:true);
        $inputs = [
            $g->Variable($K->zeros([2,4,2,2,3])),
            $g->Variable($K->zeros([2,4,2,4,3])),
        ];
        $layer->build($inputs);

        //
        // forward
        //
        $query = $K->ones(
            $inputs[0]->shape()
        );
        $value = $K->ones(
            $inputs[1]->shape()
        );
        $inputs = [$query,$value];
        $copyInputs = [$K->copy($query),$K->copy($value)];
        [$outputsVariable,$scoresVariable] = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs) {
                [$outputsVariable,$scoresVariable] = $layer->forward($inputs,
                                returnAttentionScores:true);
                return [$outputsVariable,$scoresVariable];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        $scores = $K->ndarray($scoresVariable);

        //
        $this->assertEquals([2,4,2,2,4],$scores->shape());
        $this->assertEquals([2,4,2,2,3],$outputs->shape());
        //
        // backward
        //
        $dOutputs = [
            $K->ones($outputs->shape()),
            $K->ones($scores->shape()),
        ];

        $variables = $layer->trainableVariables();
        $grads = new WeakMap();
        $copydOutputs = [];
        $copydOutputs[] = $K->copy($dOutputs[0]);
        $copydOutputs[] = $K->copy($dOutputs[1]);
        $dInputs = $outputsVariable->creator()->backward($dOutputs,$grads,$variables);
        // 2 batch
        $this->assertCount(2,$dInputs);
        $this->assertEquals([2,4,2,2,3],$dInputs[0]->shape());
        $this->assertEquals([2,4,2,4,3],$dInputs[1]->shape());

        $this->assertCount(1,$variables);
        $this->assertCount(1,$grads);
    }

}
