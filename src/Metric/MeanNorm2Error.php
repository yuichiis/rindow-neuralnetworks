<?php
namespace Rindow\NeuralNetworks\Metric;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class CategoricalAccuracy extends AbstractMetric
{
    protected string $name = 'mean_norm2_error';

    public function forward(NDArray $true, NDArray $predicts) : float
    {
        $K = $this->backend;
        $error = $K->scalar($K->nrm2($K->sub($predicts,$trues)));
        return $error/$true->size();
    }
}
