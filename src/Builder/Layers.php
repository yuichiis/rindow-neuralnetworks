<?php
namespace Rindow\NeuralNetworks\Builder;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Layer\ReLU;
use Rindow\NeuralNetworks\Layer\Sigmoid;
use Rindow\NeuralNetworks\Layer\Softmax;
use Rindow\NeuralNetworks\Layer\Dense;
use Rindow\NeuralNetworks\Layer\Flatten;
use Rindow\NeuralNetworks\Layer\Conv2D;
use Rindow\NeuralNetworks\Layer\MaxPool2D;
use Rindow\NeuralNetworks\Layer\Dropout;
use Rindow\NeuralNetworks\Layer\BatchNormalization;

class Layers
{
    protected $backend;

    public function __construct($backend)
    {
        $this->backend = $backend;
    }

    public function ReLU(array $options=null)
    {
        return new ReLU($this->backend,$options);
    }

    public function Sigmoid(array $options=null)
    {
        return new Sigmoid($this->backend,$options);
    }

    public function Softmax(array $options=null)
    {
        return new Softmax($this->backend,$options);
    }

    public function Dense(int $units, array $options=null)
    {
        return new Dense($this->backend, $units, $options);
    }

    public function Flatten(
        array $options=null)
    {
        return new Flatten($this->backend, $options);
    }

    public function Conv2D(
        int $filters, $kernel_size, array $options=null)
    {
        return new Conv2D(
            $this->backend,
            $filters,
            $kernel_size,
            $options);
    }
    
    public function MaxPool2D(
        array $options=null)
    {
        return new MaxPool2D(
            $this->backend,
            $options);
    }

    public function Dropout(float $rate,array $options=null)
    {
        return new Dropout($this->backend,$rate,$options);
    }

    public function BatchNormalization(array $options=null)
    {
        return new BatchNormalization($this->backend,$options);
    }
}
