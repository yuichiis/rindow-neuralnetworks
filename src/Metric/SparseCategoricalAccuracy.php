<?php
namespace Rindow\NeuralNetworks\Metric;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class SparseCategoricalAccuracy extends AbstractMetric
{
    protected string $name = 'sparse_categorical_accuracy';

    public function forward(NDArray $true, NDArray $predicts) : float
    {
        if(!$K->isInt($trues)) {
            throw new InvalidArgumentException('trues must be integers.');
        }
        $K = $this->backend;
        $predicts = $K->argMax($predicts,axis:-1,dtype:$true->dtype());
        $equals = $K->scalar($K->sum($K->equals($trues,$predicts)));
        return $equals/$true->size();
    }
}
