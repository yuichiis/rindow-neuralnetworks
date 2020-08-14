<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use Rindow\NeuralNetworks\Activation\FunctionFactory;
use Rindow\NeuralNetworks\Activation\Activation as ActivationInterface;

/**
 *
 */
abstract class AbstractLayerBase
{
    protected $layers = [];
    protected $inputShape;
    protected $outputShape;
    protected $statesShapes;
    protected $activation;
    protected $activationName;
    protected $params=[];
    protected $grads=[];
    protected $shapeInspection = true;

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
        if($activation instanceof ActivationInterface) {
            $this->activation = $activation;
            // for compiling lossfunction
            if($this->activationName==null){
                $this->activationName = get_class($activation);
            }
            return;
        }
        throw new InvalidArgumentException('activation function must have the Activation interface');
    }

    public function build(array $inputShape=null, array $options=null) : array
    {
        if($inputShape!==null)
            $this->inputShape = $inputShape;
        $this->outputShape = $inputShape;
        return $this->outputShape;
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

    protected function addWeights($weights,$grads=null)
    {
        if($weights instanceof Layer){
            $this->params = array_merge($this->params,$layer->getParams());
            $this->grads  = array_merge($this->grads, $layer->getGrads());
            return;
        }elseif($weights instanceof NDArray){
            if($grads==null){
                throw new InvalidArgumentException('need grads to add weights');
            }
            $this->params[]=$weights;
            $this->grads[]=$grads;
        }else{
            throw new InvalidArgumentException('invalid type to add weights');
        }
    }
    
    public function getParams() : array
    {
        return $this->params;
    }

    public function getGrads() : array
    {
        return $this->grads;
    }

    public function getConfig() : array
    {
        return [];
    }

    public function setName(string $name) : void
    {
        $this->name = $name;
    }

    public function getName() : string
    {
        return $this->name;
    }
    
    public function setShapeInspection(bool $enable)
    {
        $this->shapeInspection = $enable;
        foreach ($this->layers as $layer) {
            $layer->setShapeInspection($enable);
        }
    }

    protected function assertInputShape(NDArray $inputs)
    {
        if(!$this->shapeInspection)
            return;
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
        if(!$this->shapeInspection)
            return;
        $shape = $outputs->shape();
        $batchNum = array_shift($shape);
        if($shape!=$this->outputShape) {
            throw new InvalidArgumentException('unmatch output shape: ['.
                implode(',',$shape).'], must be ['.implode(',',$this->outputShape).']');
        }
    }

    protected function assertStatesShape(array $states=null)
    {
        if(!$this->shapeInspection)
            return;
        if($states===null)
            return;
        if($this->statesShapes===null) {
            throw new InvalidArgumentException('Uninitialized');
        }
        if(count($states)!=count($this->statesShapes)){
            throw new InvalidArgumentException('Unmatch num of status. status need '.count($this->statesShapes).' NDArray. '.count($states).'given.');
        }
        foreach($states as $idx=>$state){;
            $stateShape = $this->statesShapes[$idx];
            $shape = $inputs->shape();
            $batchNum = array_shift($shape);
            if($shape!=$stateShape) {
                $shape = $shape ? implode(',',$shape) : '';
                throw new InvalidArgumentException('unmatch shape of state '.$idx.': ['.$shape.'], must be ['.implode(',',$stateShape).']');
            }
        }
    }
    
    protected function registerLayer(LayerBase $layer,array $inputShape=null) : array
    {
        $this->layers[] = $layer;
        $outputShape = $layer->build($inputShape);
        $this->addWeights($layer);
        return $outputShape;
    }
}
