<?php
namespace Rindow\NeuralNetworks\Metric;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class BinaryAccuracy extends AbstractMetric
{
    protected string $name = 'binary_accuracy';
    protected float $threshold;

    public function __construct(
        object $backend,
        float $threshold=null,
        )
    {
        parent::__construct($backend);
        $threshold = $threshold ?? 0.5;
        $this->threshold = $threshold;
    }

    public function forward(NDArray $true, NDArray $predicts) : float
    {
        $K = $this->backend;
        $shape = $predicts->shape();
        $featuresize = array_pop($shape);

        if(!$K->isInt($trues)) {
            $trues = $K->cast($trues,dtype:int32);
        }
        $predicts = $K->cast($K->greater($predicts,$this->threshold),dtype:$trues->dtype());
        $equals = $K->scalar($K->sum($K->equals($trues,$predicts)));
        return $equals/$true->size();
    }
}
