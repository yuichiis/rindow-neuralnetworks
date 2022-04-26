<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;

class Conv2D extends AbstractConv
{
    protected $rank = 2;

    public function __construct(object $backend,int $filters, $kernel_size, array $options=null)
    {
        $leftargs = [];
        extract($this->extractArgs([
            'name'=>null,
        ],$options,$leftargs));
        $this->initName($name,'conv2d');
        parent::__construct($backend, $filters, $kernel_size, $leftargs);
    }

    protected function call(NDArray $inputs, bool $training) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        $container->status = new \stdClass();
        $outputs = $K->conv2d(
                $container->status,
                $inputs,
                $this->kernel,
                $this->bias,
                $this->strides,
                $this->padding,
                $this->data_format,
                $this->dilation_rate
        );
        if($this->activation) {
            $container->activation = new \stdClass();
            $this->activation->setStates($container->activation);
            $outputs = $this->activation->forward($outputs,$training);
        }
        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        if($this->activation) {
            $this->activation->setStates($container->activation);
            $dOutputs = $this->activation->backward($dOutputs);
        }
        $dInputs = $K->dConv2d(
            $container->status,
            $dOutputs,
            $this->dKernel,
            $this->dBias
        );
        $container->status = null;
        return $dInputs;
    }
}
