<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;

class Add extends AbstractFunction
{
    protected int $numOfInputs = 2;

    /**
    *  @param array<NDArray>  $inputs
    *  @return array<NDArray>
    */
    protected function call(array $inputs) : array
    {
        $container = $this->container();
        $container->inputs = $inputs;

        $output = $this->backend->add($inputs[0],$inputs[1]);
        return [$output];
    }

    /**
    *  @param array<NDArray>  $dOutputs
    *  @return array<NDArray>
    */
    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        [$x0, $x1] = $container->inputs;

        $dx0 = $dx1 = $dOutputs[0];

        // for broadcasted inputs
        if($x0->ndim() != $dx0->ndim()) {
            $dx0 = $K->sum($dx0, axis:0);
        }
        if($x1->ndim() != $dx1->ndim()) {
            $dx1 = $K->sum($dx1, axis:0);
        }
        return [$dx0, $dx1];
    }
}
