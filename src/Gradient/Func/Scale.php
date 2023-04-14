<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use Rindow\NeuralNetworks\Gradient\Scalar as ScalarInterface;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;
use Rindow\NeuralNetworks\Gradient\Core\Scalar;
use Interop\Polite\Math\Matrix\NDArray;

class Scale extends AbstractFunction
{
    protected $numOfInputs = 2;

    protected function preprocess(array $inputs) : array
    {
        if(is_numeric($inputs[0])) {
            $inputs[0] = new Scalar($inputs[0]);
        }
        return $inputs;
    }

    /**
    *  @param array<NDArray>  $inputs
    *       inputs
    *  @return array<NDArray>
    *       outputs
    */
    protected function call(array $inputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $container->inputs = $inputs;

        $alpha = $inputs[0];
        $array = $inputs[1];

        if($alpha instanceof ScalarInterface) {
            $alpha = $alpha->value();
        } elseif($alpha instanceof NDArray) {
            if($alpha->ndim()!=0) {
                throw new InvalidArgumentException('arg #1 must not be scalar.');
            }
            $alpha = $K->scalar($alpha);
        } else {
            if(is_object($alpha)) {
                $type = get_class($alpha);
            } else {
                $type = gettype($alpha);
            }
            throw new InvalidArgumentException("arg #1 is invalid data type.: ".$type);
        }
        $output = $K->scale($alpha,$array);
        return [$output];
    }

    /**
    *  @param array<NDArray>  $dOutputs
    *       difference outputs
    *  @return array<NDArray>
    *       difference inputs
    */
    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        [$alpha, $array] = $container->inputs;

        if($alpha instanceof ScalarInterface) {
            $alpha = $alpha->value();
            $dAlpha = new Scalar(0);
        } elseif($alpha instanceof NDArray) {
            if($alpha->ndim()!=0) {
                throw new InvalidArgumentException('arg #1 must not be scalar.');
            }
            $alpha = $K->scalar($alpha);
            $dAlpha = $K->sum($dOutputs[0]);
            if(!($dAlpha instanceof NDArray)) {
                $dAlpha = $K->array($dAlpha);
            }
        }

        $dInputs = $K->scale($alpha,$dOutputs[0]);
        
        return [$dAlpha, $dInputs];
    }
}
