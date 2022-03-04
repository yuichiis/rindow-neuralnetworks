<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

/**
 *
 */
abstract class AbstractRNNCell extends AbstractLayerBase implements RNNCell
{
    abstract protected function call(NDArray $inputs, array $states, bool $training, object $calcState, array $options=null) : array;
    abstract protected function differentiate(NDArray $dOutputs, array $dStates, object $calcState) : array;

    public function getParams() : array
    {
        if($this->bias) {
            return [$this->kernel,$this->recurrentKernel,$this->bias];
        } else {
            return [$this->kernel,$this->recurrentKernel];
        }
    }

    public function getGrads() : array
    {
        if($this->bias) {
            return [$this->dKernel,$this->dRecurrentKernel,$this->dBias];
        } else {
            return [$this->dKernel,$this->dRecurrentKernel];
        }
    }

    public function reverseSyncCellWeightVariables(array $weights) : void
    {
        if($this->useBias) {
            $this->kernel = $weights[0]->value();
            $this->recurrentKernel = $weights[1]->value();
            $this->bias = $weights[2]->value();
        } else {
            $this->kernel = $weights[0]->value();
            $this->recurrentKernel = $weights[1]->value();
        }
    }

    final public function forward(NDArray $inputs, array $states, bool $training, object $calcState, array $options=null) : array
    {
        $this->assertInputShape($inputs,'forward');

        [$outputs,$states] = $this->call($inputs,$states,$training,$calcState,$options);

        $this->assertOutputShape($outputs,'forward');
        return [$outputs,$states];
    }

    final public function backward(NDArray $dOutputs, array $dStates, object $calcState) : array
    {
        $this->assertOutputShape($dOutputs,'backward');

        [$dInputs,$dStates] = $this->differentiate($dOutputs,$dStates,$calcState);

        $this->assertInputShape($dInputs,'backward');
        return [$dInputs,$dStates];
    }
}
