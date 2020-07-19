<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class Conv2D extends AbstractImage implements Layer
{
    use GenericUtils;
    protected $rank = 2;
    protected $backend;
    protected $filters;
    protected $kernel_size;
    protected $strides;
    protected $padding;
    protected $data_format;
    protected $activation;
    protected $useBias;
    protected $kernelInitializer;
    protected $biasInitializer;

    protected $kernel;
    protected $bias;
    protected $dKernel;
    protected $dBias;
    protected $status;
    

    public function __construct($backend,int $filters, $kernel_size, array $options=null)
    {
        extract($this->extractArgs([
            'strides'=>[1, 1],
            'padding'=>"valid",
            'data_format'=>null,
            # 'dilation_rate'=>[1, 1],
            'groups'=>1,
            'activation'=>null,
            'use_bias'=>true,
            'kernel_initializer'=>"sigmoid_normal",
            'bias_initializer'=>"zeros",
            'kernel_regularizer'=>null,
            'bias_regularizer'=>null,
            'activity_regularizer'=>null,
            'kernel_constraint'=>null,
            'bias_constraint'=>null,
            
            'input_shape'=>null,
            'activation'=>null,
            'use_bias'=>true,

        ],$options));
        $this->backend = $K = $backend;
        $kernel_size=$this->normalizeFilterSize($kernel_size,'kernel_size',
            null,true);
        $strides=$this->normalizeFilterSize($strides,'strides',[1,1]);
        $this->kernel_size = $kernel_size;
        $this->filters = $filters;
        $this->strides = $strides;
        $this->padding = $padding;
        $this->data_format = $data_format;
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
        $kernel_size = $this->kernel_size;
        $outputShape = 
            $K->calcConv2dOutputShape(
                $this->inputShape,
                $this->kernel_size,
                $this->strides,
                $this->padding,
                $this->data_format
            );
        array_push($outputShape,$this->filters);
        
        $channels = $this->getChannels();
        array_push($kernel_size,
            $channels);
        array_push($kernel_size,
            $this->filters);
        if($sampleWeights) {
            $this->kernel = $sampleWeights[0];
            $this->bias = $sampleWeights[1];
        } else {
            $this->kernel = $kernelInitializer($kernel_size);
            $this->bias = $biasInitializer([$this->filters]);
        }
        $this->dKernel = $K->zerosLike($this->kernel);
        $this->dBias = $K->zerosLike($this->bias);
        $this->outputShape = $outputShape;
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
            'filters' => $this->filters,
            'kernel_size' => $this->kernel_size,
            'options' => [
                'strides' => $this->strides,
                'padding' => $this->padding,
                'data_format' => $this->data_format,
                'input_shape'=>$this->inputShape,
                'kernel_initializer' => $this->kernelInitializerName,
                'bias_initializer' => $this->biasInitializerName,
            ]
        ]);
    }
    
    protected function call(NDArray $inputs, bool $training) : NDArray
    {
        $K = $this->backend;
        $this->status = new \stdClass();
        $outputs = $K->conv2d(
                $this->status,
                $inputs,
                $this->kernel,
                $this->bias,
                $this->strides,
                $this->padding,
                $this->data_format
        );
        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        $dInputs = $K->dConv2d(
            $this->status,
            $dOutputs,
            $this->dKernel,
            $this->dBias
        );
        $this->status = null;
        return $dInputs;
    }
}
