<?php
namespace Rindow\NeuralNetworks\Activation;

abstract class AbstractActivation implements Activation
{
    public function __construct($backend)
    {
        $this->backend = $backend;
    }
}

