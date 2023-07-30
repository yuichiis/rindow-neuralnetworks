<?php
namespace Rindow\NeuralNetworks\Metric;

use Interop\Polite\Math\Matrix\NDArray;

abstract class AbstractMetric implements Metric
{
    protected object $backend;

    public function __construct(
        object $backend,
        )
    {
        $this->backend = $backend;
    }

    public function name() : string
    {
        return $this->name;
    }

    public function __invoke(...$args) : mixed
    {
        [$true,$predicts] = $args;
        return $this->forward($true, $predicts);
    }
}
