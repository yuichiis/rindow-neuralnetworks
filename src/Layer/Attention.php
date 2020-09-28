<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class Activation extends AbstractLayerBase
{
    use GenericUtils;
    protected $backend;

    public function __construct($backend, array $options=null)
    {
        extract($this->extractArgs([
            'input_shape'=>null,
        ],$options));
        $this->backend = $K = $backend;
        $this->inputShape = $input_shape;
    }

    public function getConfig() : array
    {
        return [
            'options' => [
                'input_shape'=>$this->inputShape,
            ],
        ];
    }
    
    public function forward(array $inputs, bool $training) : NDArray
    {
        if(count($inputs)!=2||count($inputs)!=3) {
            throw new InvalidArgumentException('Must have 2 or 3 arguments');
        }
        return $this->call($inputs,$training);
    }
    
    protected function backword(NDArray $dOutputs) : array
    {
        return $this->differentiate($dOutputs);
    }

    protected function call(array $inputs, bool $training) : array
    {
        $K = $this->backend;
        $query = $inputs[0];
        $value = $inputs[1];
        if(count($inputs)==3) {
            $key = $inputs[2];
        } else {
            $key = $inputs[1];
        }

        $scores = $K->gemm($query, $key, null,null,null,null, $tranB=true);
        $attentionWeight = $K->softmax($scores);

        $contextVector = $K->gemm($attentionWeight, $value);
        $this->value = $value;
        $this->attentionWeight = $attentionWeight;
        return $contextVector;
    }

    protected function differentiate(NDArray $dOutputs) : array
    {
        $K = $this->backend;
        $dValue = $K->gemm($dOutputs,$this->attentionWeight);
        $dAttentionWeight = $K->gemm($this->value,$dOutputs);
        $dScore = $K->dSoftmax($scores,$dAttentionWeight);
        $dkey = $K->gemm($dScore,$this->query);
        
        return $dInputs;
    }
}
