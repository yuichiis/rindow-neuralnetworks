<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class Activation extends AbstractLayer implements Layer
{
    use GenericUtils;
    protected $backend;

    public function __construct($backend, $activation,array $options=null)
    {
        extract($this->extractArgs([
            'input_shape'=>null,
        ],$options));
        $this->backend = $K = $backend;
        $this->inputShape = $input_shape;
        $this->setActivation($activation);
    }
    
    public function getConfig() : array
    {
        return [
            'activation'=>$this->activationName,
            'options' => [
                'input_shape'=>$this->inputShape,
            ],
        ];
    }

    protected function call(NDArray $inputs, bool $training) : NDArray
    {
        return $inputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        return $dOutputs;
    }
}
