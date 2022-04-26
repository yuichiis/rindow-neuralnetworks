<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;

class AveragePooling1D extends AbstractPooling
{
    protected $rank = 1;
    protected $pool_mode = 'avg';

    public function __construct(object $backend,array $options=null)
    {
        $leftargs = [];
        extract($this->extractArgs([
            'name'=>null,
        ],$options,$leftargs));
        $this->initName($name,'averagepooling1d');
        parent::__construct($backend, $leftargs);
    }

    protected function call(NDArray $inputs, bool $training) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        $container->status = new \stdClass();
        $outputs = $K->pool1d(
                $container->status,
                $inputs,
                $this->poolSize,
                $this->strides,
                $this->padding,
                $this->data_format,
                $this->dilation_rate,
                $this->pool_mode
        );
        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        $dInputs = $K->dPool1d(
            $container->status,
            $dOutputs
        );
        $container->status = null;
        return $dInputs;
    }
}
