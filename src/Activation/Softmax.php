<?php
namespace Rindow\NeuralNetworks\Activation;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class Softmax extends AbstractActivation
{
    protected function call(NDArray $inputs, bool $training=null, NDArray $mask=null) : NDArray
    {
        $K = $this->backend;
        if(!$mask) {
            $outputs = $K->softmax($inputs);
            $this->states->outputs = $outputs;
            return $outputs;
        }

        //
        // masked softmax
        //
        $ndim = $mask->ndim();
        $orignalInputShape = $inputs->shape();
        $batchShape = $orignalInputShape;
        $inputShape = array_splice($batchShape,-$ndim);
        echo "inputs=".$K->localMatrixOperator()->shapeToString($inputs->shape())."\n";
        echo "mask=".$K->localMatrixOperator()->shapeToString($mask->shape())."\n";
        echo "ndim=$ndim\n";
        echo "batchShape=".$K->localMatrixOperator()->shapeToString($batchShape)."\n";
        echo "inputShape=".$K->localMatrixOperator()->shapeToString($inputShape)."\n";
        if($inputShape!=$mask->shape()) {
            throw new InvalidArgumentException('unmatch shape of inputs and mask: '.
                'inputs=('.implode(',',$inputs->shape()).'), '.
                'mask=('.implode(',',$mask->shape()).')'
            );
        }

        echo $K->localMatrixOperator()->toString($mask,indent:true)."\n";
        echo "mask=".$K->localMatrixOperator()->shapeToString($mask->shape())."\n";
        //$outputs = $K->softmax($inputs);
        $nums = $K->sum($mask,axis:1);
        echo "nums=".$K->localMatrixOperator()->shapeToString($nums->shape())."\n";
        echo $K->localMatrixOperator()->toString($nums,indent:true)."\n";
        $masked_inputs = $K->mul($inputs,$mask);
        echo "masked_inputs=".$K->localMatrixOperator()->shapeToString($masked_inputs->shape())."\n";
        $batches = (int)array_product($batchShape);

        ////////////////
        // exp_diff = exp(inputs-max)
        // sum_exp = sum( exp_diff*mask )
        // softmax = exp_diff / sum_exp
        /////////////////
        [$rows,$cols] = $inputShape;
        $inputs = $inputs->reshape([$batches*$rows,$cols]);
        echo "flat_inputs=".$K->localMatrixOperator()->shapeToString($inputs->shape())."\n";
        $maxes = $K->max($inputs,axis:-1);
        echo "maxes=".$K->localMatrixOperator()->shapeToString($maxes->shape())."\n";
        $expDiff = $K->exp($K->sub($inputs,$maxes,trans:true));
        $expDiff = $expDiff->reshape([$batches,$rows,$cols]);
        $sumExp = $K->sum($K->mul($expDiff,$mask),axis:-1);
        $expDiff = $expDiff->reshape([$batches*$rows,$cols]);
        $sumExp = $sumExp->reshape([$batches*$rows]);
        $outputs = $K->div($expDiff, $sumExp, trans:true);
        $outputs = $outputs->reshape($orignalInputShape);

        $this->states->outputs = $outputs;
        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        return $K->dSoftmax($dOutputs, $this->states->outputs);
    }
}
