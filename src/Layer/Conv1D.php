<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;

class Conv1D extends AbstractConv implements Layer
{
    protected $rank = 1;
    
    protected function call(NDArray $inputs, bool $training) : NDArray
    {
        $K = $this->backend;
        $this->status = new \stdClass();
        $outputs = $K->conv1d(
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
        $dInputs = $K->dConv1d(
            $this->status,
            $dOutputs,
            $this->dKernel,
            $this->dBias
        );
        $this->status = null;
        return $dInputs;
    }
}
