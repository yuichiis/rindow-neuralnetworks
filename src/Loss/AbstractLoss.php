<?php
namespace Rindow\NeuralNetworks\Loss;

use Interop\Polite\Math\Matrix\NDArray;
//use Rindow\NeuralNetworks\Activation\Activation;
use Rindow\NeuralNetworks\Gradient\Core\Variable;
use Rindow\NeuralNetworks\Gradient\Core\GradientTape;
use Rindow\NeuralNetworks\Gradient\Core\GradientUtils;
use Rindow\NeuralNetworks\Support\GenericUtils;
use InvalidArgumentException;
use DomainException;
use ArrayAccess;

abstract class AbstractLoss //implements Loss
{
    use GradientUtils;
    protected $generation;
    protected $inputsVariables;
    protected $outputsVariables;

    abstract protected function call(NDArray $trues, NDArray $predicts) : float;
    abstract protected function differentiate(array $dOutputs, ArrayAccess $grads=null, array $oidsToCollect=null) : array;

    /*
    *  dinamic step interfaces
    */
    /**
    *  @return int
    */
    public function generation() : int
    {
        return $this->generation;
    }
    /**
    *  @return array<Variable>
    */
    public function inputs()
    {
        return $this->inputsVariables;
    }

    /**
    *  @return array<Variable>
    */
    public function outputs()
    {
        return $this->outputsVariables;
    }

    public function backward(array $dOutputs, ArrayAccess $grads=null, array $oidsToCollect=null) : array
    {
        return $this->differentiate($dOutputs, $grads, $oidsToCollect);
    }

    public function __invoke(...$args)
    {
        return $this->forward(...$args);
    }

    /**
    *  @param array<Variable>  $inputs
    *       inputs
    *  @return array<Variable>
    *       outputs
    */
    public function forward(NDArray $trues, NDArray $predicts) : Variable
    {
        $K = $this->backend;
        [$predicts,$rawPredicts] = $this->packAndUnpackVariable($this->backend,$predicts);
        $session = $this->preGradientProcessOnSession([$predicts]);
        $session->begin();
        try {
            $loss = $this->call($trues,$rawPredicts);
            $rawOutputs = $K->array($loss,$rawPredicts->dtype());
        } finally {
            $session->end();
        }
        $outputs = $this->postGradientProcessOnSession(
            $this->backend, $session, [$predicts], [$rawOutputs]);
        return $outputs[0];
    }
}
