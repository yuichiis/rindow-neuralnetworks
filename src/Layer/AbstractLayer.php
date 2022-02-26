<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use LogicException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\Variable;
use Rindow\NeuralNetworks\Gradient\Core\Undetermined;
use Rindow\NeuralNetworks\Gradient\Core\UndeterminedNDArray;
use Rindow\NeuralNetworks\Gradient\Core\GradientTape;
use Rindow\NeuralNetworks\Gradient\Core\GradientUtils;
use Rindow\NeuralNetworks\Model\BuildContext;

/**
 *
 */
abstract class AbstractLayer extends AbstractLayerBase
{
    use GradientUtils;
    abstract protected function call(NDArray $inputs, bool $training) : NDArray;
    abstract protected function differentiate(NDArray $dOutputs) : NDArray;

    final public function forward(object $inputs, bool $training)
    {
        if(BuildContext::$build) {
            return $this->build($inputs);
        }
        $this->assertInputShape($inputs,'forward');

        $outputs = $this->call($inputs, $training);

        $this->assertOutputShape($outputs,'forward');
        return $outputs;
    }

    /**
    *  @param  array<NDArray> $dOutputs
    *  @return array<NDArray>
    */
    final public function backward(array $dOutputs, array &$grads=null, array $oidsToCollect=null) : array
    {
        if(count($dOutputs)!=1) {
            throw new InvalidArgumentException('dOutputs must be list containing one NDArray');
        }
        $dOutputs = $dOutputs[0];
        if(!($dOutputs instanceof NDArray)) {
            throw new InvalidArgumentException('dOutputs must be list containing one NDArray');
        }
        $this->assertOutputShape($dOutputs,'backward');
        $dInputs = $this->differentiate($dOutputs);
        $this->assertInputShape($dInputs,'backward');
        $this->collectGradients($grads,$oidsToCollect);
        return [$dInputs];
    }

    /**
    *  @param Variable  $inputs
    *       inputs
    *  @return array<Variable>
    *       outputs
    */
    public function __invoke($inputs, $training)
    {
        $outputs = null;
        if($this->outputShape==null) {
            $inputShape = null;
            $creator = $inputs->creator();
            if($creator) {
                $inputShape = $inputs;
            }
            $outputs = $this->build($inputShape);
        }
        if($inputs instanceof Undetermined) {
            if($outputs===null) {
                throw new InvalidArgumentException('Undetermined is found in second calling.');
            }
            return $outputs;
        }
        $inputs = $this->packVariable($this->backend, $inputs);
        $training = $this->packVariable($this->backend, $training);

        $session = $this->preGradientProcessOnSession([$inputs],['training'=>$training]);
        $session->begin();
        try {
            $outputs = $this->forward($inputs->value(),$training->value());
        } catch(Throwable $e) {
            $session->end();
            throw $e;
        }
        $session->end();
        $outputs = $this->postGradientProcessOnSession(
            $this->backend, $session, [$inputs], [$outputs]);
        return $outputs[0];
    }

    /**
     * Call from SessionFunc in compiled graph
     */
    public function _rawCall(array $inputs,array $options)
    {
        $training = $options['training'] ?? false;
        $outputs = $this->call($inputs[0],$training);
        return [$outputs];
    }
}
