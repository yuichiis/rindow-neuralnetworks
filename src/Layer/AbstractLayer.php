<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use Rindow\NeuralNetworks\Activation\FunctionFactory;

/**
 *
 */
abstract class AbstractLayer
{
    abstract protected function call(NDArray $inputs, bool $training) : NDArray;
    abstract protected function differentiate(NDArray $dOutputs) : NDArray;

    protected $inputShape;
    protected $outputShape;
    protected $activation;
    protected $activationName;

    public function getActivation()
    {
        return $this->activation;
    }
    
    public function setActivation(
        $activation) : void
    {
        if($activation==null){
            return;
        }
        if(is_string($activation)) {
            $this->activation = FunctionFactory::factory($this->backend,$activation);
            $this->activationName = $activation;
            return;
        }
        if($activation instanceof Activation) {
            $this->activation = $activation;
            // for compiling lossfunction
            if($this->activationName==null){
                $this->activationName = get_class($activation);
            }
            return;
        }
        throw new InvalidArgumentException('activation function must have the Activation interface');
    }

    public function build(array $inputShape=null, array $options=null)
    {
        if($inputShape!==null)
            $this->inputShape = $inputShape;
        $this->outputShape = $inputShape;
    }

    protected function normalizeInputShape(array $inputShape=null) : array
    {
        if($inputShape===null)
            $inputShape = $this->inputShape;
        if($this->inputShape===null)
            $this->inputShape = $inputShape;
        if($this->inputShape!==$inputShape) {
            throw new InvalidArgumentException(
                'Input shape is inconsistent: ['.implode(',',$this->inputShape).
                '] and ['.implode(',',$inputShape).']');
        } elseif($inputShape===null) {
            throw new InvalidArgumentException('Input shape is not defined');
        }
        $this->inputShape = $inputShape;
        return $inputShape;
    }

    public function outputShape() : array
    {
        return $this->outputShape;
    }

    public function getParams() : array
    {
        return [];
    }

    public function getGrads() : array
    {
        return [];
    }

    public function getConfig() : array
    {
        return [
            'activation'=>$this->activationName;
            //'input_shape' => $this->inputShape,
            //'output_shape' => $this->outputShape,
        ];
    }

    public function setName(string $name) : void
    {
        $this->name = $name;
    }

    public function getName() : string
    {
        return $this->name;
    }

    protected function assertInputShape(NDArray $inputs)
    {
        if($this->inputShape===null) {
            throw new InvalidArgumentException('Uninitialized');
        }
        $shape = $inputs->shape();
        $batchNum = array_shift($shape);
        if($shape!=$this->inputShape) {
            $shape = $shape ? implode(',',$shape) : '';
            throw new InvalidArgumentException('unmatch input shape: ['.$shape.'], must be ['.implode(',',$this->inputShape).']');
        }
    }

    protected function assertOutputShape(NDArray $outputs)
    {
        $shape = $outputs->shape();
        $batchNum = array_shift($shape);
        if($shape!=$this->outputShape) {
            throw new InvalidArgumentException('unmatch output shape: ['.
                implode(',',$shape).'], must be ['.implode(',',$this->outputShape).']');
        }
    }

    final public function forward(NDArray $inputs, bool $training) : NDArray
    {
        $this->assertInputShape($inputs);

        $outputs = $this->call($inputs, $training);

        $this->assertOutputShape($outputs);
        if($this->activation)
            $outputs = $this->activation->call($outputs,$training);
        return $outputs;
    }

    final public function backward(NDArray $dOutputs) : NDArray
    {
        $this->assertOutputShape($dOutputs);
        if($this->activation)
            $dOutputs = $this->activation->differentiate($dOutputs);

        $dInputs = $this->differentiate($dOutputs);

        $this->assertInputShape($dInputs);
        return $dInputs;
    }
}
