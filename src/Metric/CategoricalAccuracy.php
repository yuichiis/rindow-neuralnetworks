<?php
namespace Rindow\NeuralNetworks\Metric;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class CategoricalAccuracy extends AbstractMetric
{
    protected string $name = 'categorical_accuracy';

    public function forward(NDArray $true, NDArray $predicts) : float
    {
        $K = $this->backend;
        $shape = $predicts->shape();
        $featuresize = array_pop($shape);
        if($$featuresize>2147483648) {
            $dtype = NDArray::int64;
        } else {
            $dtype = NDArray::int32;
        }
        $trues = $K->argMax($predicts,axis:-1,dtype:$dtype);
        $predicts = $K->argMax($predicts,axis:-1,dtype:$dtype);
        $equals = $K->scalar($K->sum($K->equals($trues,$predicts)));
        return $equals/$true->size();
    }
}
