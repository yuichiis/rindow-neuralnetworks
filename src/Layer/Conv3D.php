<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;

class Conv3D extends AbstractConv implements Layer
{
    protected $rank = 3;

    protected function call(NDArray $inputs, bool $training) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        $container->status = new \stdClass();
        $outputs = $K->conv3d(
                $container->status,
                $inputs,
                $this->kernel,
                $this->bias,
                $this->strides,
                $this->padding,
                $this->data_format,
                $this->dilation_rate
        );
        if($this->activation)
            $outputs = $this->activation->forward($outputs,$training);
        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        if($this->activation)
            $dOutputs = $this->activation->backward($dOutputs);
        $dInputs = $K->dConv3d(
            $container->status,
            $dOutputs,
            $this->dKernel,
            $this->dBias
        );
        $container->status = null;
        return $dInputs;
    }
}
