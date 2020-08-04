<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class Dense extends AbstractLayer implements Layer
{
    use GenericUtils;
    protected $backend;
    protected $units;
    protected $activation;
    protected $useBias;
    protected $kernelInitializer;
    protected $biasInitializer;

    protected $kernel;
    protected $bias;
    protected $dKernel;
    protected $dBias;
    protected $inputs;

    public function __construct($backend,int $units, array $options=null)
    {
        extract($this->extractArgs([
            'input_shape'=>null,
            'activation'=>null,
            'use_bias'=>true,
            'kernel_initializer'=>'sigmoid_normal',
            'bias_initializer'=>'zeros',
            //'kernel_regularizer'=>null, 'bias_regularizer'=>null,
            //'activity_regularizer'=null,
            //'kernel_constraint'=null, 'bias_constraint'=null,
        ],$options));
        $this->backend = $K = $backend;
        $this->units = $units;
        $this->inputShape = $input_shape;
        $this->activation = $activation;
        $this->kernelInitializer = $K->getInitializer($kernel_initializer);
        $this->biasInitializer   = $K->getInitializer($bias_initializer);
        $this->kernelInitializerName = $kernel_initializer;
        $this->biasInitializerName = $bias_initializer;
    }

    public function build(array $inputShape=null, array $options=null) : void
    {
        extract($this->extractArgs([
            'sampleWeights'=>null,
        ],$options));
        $K = $this->backend;
        $kernelInitializer = $this->kernelInitializer;
        $biasInitializer = $this->biasInitializer;

        $inputShape = $this->normalizeInputShape($inputShape);
        //if(count($inputShape)!=1) {
        //    throw new InvalidArgumentException(
        ///        'Unsuppored input shape: ['.implode(',',$inputShape).']');
        //}
        $shape = $inputShape;
        $this->inputDim=array_pop($shape);
        if($sampleWeights) {
            $this->kernel = $sampleWeights[0];
            $this->bias = $sampleWeights[1];
        } else {
            $this->kernel = $kernelInitializer([$this->inputDim,$this->units],$this->inputDim);
            $this->bias = $biasInitializer([$this->units]);
        }
        $this->dKernel = $K->zerosLike($this->kernel);
        $this->dBias = $K->zerosLike($this->bias);
        array_push($shape,$this->units);
        $this->outputShape = $shape;
    }

    public function getParams() : array
    {
        return [$this->kernel,$this->bias];
    }

    public function getGrads() : array
    {
        return [$this->dKernel,$this->dBias];
    }

    public function getConfig() : array
    {
        return array_merge(parent::getConfig(),[
            'units' => $this->units,
            'options' => [
                'input_shape'=>$this->inputShape,
                'kernel_initializer' => $this->kernelInitializerName,
                'bias_initializer' => $this->biasInitializerName,
            ]
        ]);
    }

    protected function call(NDArray $inputs, bool $training) : NDArray
    {
        $K = $this->backend;
        $shape = $inputs->shape();
        $this->origInputsShape = $shape;
        $inputDim=array_pop($shape);
        $inputSize=array_product($shape);
        $this->inputs = $inputs->reshape([$inputSize,$inputDim]);
        $outputs = $K->batch_gemm($this->inputs, $this->kernel,1.0,1.0,$this->bias);
        array_push($shape,$this->units);
        return $outputs->reshape($shape);
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        $dInputs = $K->zerosLike($this->inputs);
        $shape = $dOutputs->shape();
        $outputDim=array_pop($shape);
        $outputSize=array_product($shape);
        $dOutputs=$dOutputs->reshape([$outputSize,$outputDim]);
        $K->gemm($dOutputs, $this->kernel,1.0,0.0,$dInputs,false,true);

        // update params
        $K->gemm($this->inputs, $dOutputs,1.0,0.0,$this->dKernel,true,false);
        $K->copy($K->sum($dOutputs, $axis=0),$this->dBias);

        return $dInputs->reshape($this->origInputsShape);
    }
}
