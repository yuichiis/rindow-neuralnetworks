<?php
namespace Rindow\NeuralNetworks\Loss;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use DomainException;

class MeanSquaredError extends AbstractGradient implements Loss
{
    protected $backend;
    //protected $trues;
    //protected $predicts;

    public function __construct(object $backend,array $options=null)
    {
        //extract($this->extractArgs([
        //],$options));
        $this->backend = $backend;
    }

    public function getConfig() : array
    {
        return [
        ];
    }

    public function forward(NDArray $trues, NDArray $predicts) : float
    {
        $K = $this->backend;
        $container = $this->container();
        //$this->assertOutputShape($predicts);
        //if($trues->ndim()!=2) {
        //    throw new InvalidArgumentException('categorical\'s "trues" must be shape of [batchsize,1].');
        //}
        if($trues->shape()!=$predicts->shape())
            throw new InvalidArgumentException('unmatch shape of trues and predicts results');
        $container->trues = $trues;
        $container->predicts = $predicts;
        $loss = $K->meanSquaredError($trues, $predicts);
        return $loss;
    }

    public function backward(array $dOutputs, array &$grads=null, array $oidsToCollect=null) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $dInputs = $K->dMeanSquaredError($container->trues, $container->predicts);
        return [$dInputs];
    }

    public function accuracy(
        NDArray $trues, NDArray $predicts) : float
    {
        $K = $this->backend;
        if($trues->shape()!=$predicts->shape())
            throw new InvalidArgumentException('unmatch shape of trues and predicts results');
        // calc accuracy
        $shape=$predicts->shape();
        if(count($shape)>=2) {
            if($predicts->shape()[1]>2147483648) {
                $dtype = NDArray::int64;
            } else {
                $dtype = NDArray::int32;
            }
            $predicts = $K->argmax($predicts, $axis=1,$dtype);
            $trues = $K->argmax($trues, $axis=1,$dtype);
            $sum = $K->sum($K->equal($trues, $predicts));
        } else {
            $sum = $K->nrm2($K->sub($predicts,$trues));
        }
        $sum = $K->scalar($sum);
        $accuracy = $sum / (float)$trues->shape()[0];
        return $accuracy;
    }
}
