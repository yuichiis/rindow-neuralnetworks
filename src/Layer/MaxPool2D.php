<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class MaxPool2D extends AbstractLayer implements Layer
{
    use GenericUtils;
    protected $backend;
    protected $pool_size;
    protected $strides;
    protected $padding;
    protected $data_format;
    protected $status;
    protected $status;
    protected $pool_mode;
    

    public function __construct($backend,array $options=null)
    {
        extract($this->extractArgs([
            'pool_size'=>[2,2]
            'strides'=>null,
            'padding'=>"valid",
            'data_format'=>null,
            # 'dilation_rate'=>[1, 1],
            'input_shape'=>null,
        ],$options));
        $this->backend = $backend;
        if($pool_size===null) {
            $pool_size = [2,2];
        } elseif(is_int($pool_size)) {
            $pool_size = [ $pool_size,$pool_size];
        } elseif(!is_array($pool_size)) {
            throw new InvalidArgumentException('pool_size must be integer or array of integer');
        }
        if($strides===null) {
            $strides = $pool_size;
        } elseif(is_int($strides)) {
            $strides = [$strides,$strides];
        } else {
            throw new InvalidArgumentException('strides must be integer or array of integer');
        }
        $this->pool_size = $pool_size;
        $this->strides = $strides;
        $this->padding = $padding;
        $this->data_format = $data_format;
        $this->pool_mode = 'max';
        $this->inputShape = $input_shape;
    }

    public function build(array $inputShape=null, array $options=null) : void
    {
        extract($this->extractArgs([
            'sampleWeights'=>null,
        ],$options));
        $K = $this->backend;

        if($inputShape===null)
            $inputShape = $this->inputShape;
        if($this->inputShape===null)
            $this->inputShape = $inputShape;
        if($this->inputShape!==$inputShape)
        {
            throw new InvalidArgumentException(
                'Input shape is inconsistent: ['.implode(',',$this->inputShape).
                '] and ['.implode(',',$inputShape).']');
        } elseif($inputShape===null) {
            throw new InvalidArgumentException('Input shape is not defined');
        }
        if(count($inputShape)!=3) {
            throw new InvalidArgumentException(
                'Unsuppored input shape: ['.implode(',',$inputShape).']');
        }
        if($this->data_format==null||
           $this->data_format=='channels_last') {
            $channels = $inputShape[2];
        } elseif($this->data_format=='channels_first') {
            $channels = $inputShape[0];
        } else {
            throw new InvalidArgumentException('data_format is invalid');
        }
        $outputShape = 
            $K->calcConv2dOutputShape(
                $this->inputShape,
                $this->pool_size,
                $this->strides,
                $this->padding,
                $this->data_format
            );
        array_push($outputShape,$channels);
        $this->inputShape = $inputShape;
        $this->outputShape = $outputShape;
    }

    public function getParams() : array
    {
        return [];
    }

    public function getGrads() : array
    {
        return [];
    }

    public function getConfig() : array
    {
        return array_merge(parent::getConfig(),[
            'options' => [
                'pool_size' => $this->pool_size,
                'strides' => $this->strides,
                'padding' => $this->padding,
                'data_format' => $this->data_format,
                'input_shape'=>$this->inputShape,
            ]
        ]);
    }
    
    protected function call(NDArray $inputs, bool $training) : NDArray
    {
        $K = $this->backend;
        $this->status = new Class();
        $outputs = $K->pool2d(
                $this->status,
                $inputs,
                $this->poolSize,
                $this->strides,
                $this->padding,
                $this->data_format,
                $this->pool_mode
        );
        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        $dInputs = $K->dPool2d(
            $this->status,
            $dOutputs
        );
        $this->status = null;
        return $dInputs;
    }
}
