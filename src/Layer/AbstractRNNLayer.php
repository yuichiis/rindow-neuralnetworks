<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use Rindow\NeuralNetworks\Gradient\Core\Variable;
use Rindow\NeuralNetworks\Gradient\Core\Undetermined;
use Rindow\NeuralNetworks\Gradient\Core\UndeterminedNDArray;
use Rindow\NeuralNetworks\Gradient\Core\GradientTape;
use Rindow\NeuralNetworks\Gradient\Core\GradientUtils;
use Rindow\NeuralNetworks\Model\BuildContext;

/**
 *
 */
abstract class AbstractRNNLayer extends AbstractLayerBase implements RNNLayer
{
    use GradientUtils;
    abstract protected function numOfOutputStates($options);

    protected $initialStates; // the statefull variable is not in container
    //protected $calcStates;
    //protected $origInputsShape;
    //protected $enableInitialStates;

    final public function forward(object $inputs, bool $training, array $initialStates=null,array $options=null)
    {
        $variables = [$inputs];
        if($initialStates!==null) {
            $variables = array_merge($variables,$initialStates);
        }

        if(BuildContext::$build) {
            $results = $this->build($variables,$options);
            if(is_array($results)) {
                $outputs = array_shift($results);
                return [$outputs,$results];
            } else {
                return $outputs;
            }
        }
        $this->assertInputShape($inputs,'forward');
        $this->assertStatesShape($initialStates,'forward');
        $results = $this->call($variables,$training);
        $states = $results;
        $outputs = array_shift($states);
        if(count($states)>0) {
            $this->assertStatesShape($states,'forward');
        }
        $this->assertOutputShape($outputs,'forward');
        if(count($states)>0) {
            return [$outputs,$states];
        }
        return $outputs;
    }

    /**
    *  @param  array<NDArray> $dOutputs
    *  @return array<NDArray>
    */
    final public function backward(array $dOutputs,array &$grads=null,array $oidsToCollect=null) : array
    {
        if(!$this->shapeInspection) {
            $tmpdStates = $dOutputs;
            $tmpdOutputs = array_shift($tmpdStates);
            if(!($tmpdOutputs instanceof NDArray)) {
                throw new InvalidArgumentException('dOutputs must be list of NDArray');
            } elseif(count($tmpdStates)==0) {
                $tmpdStates = null;
            }
            $this->assertOutputShape($tmpdOutputs,'backward');
            $this->assertStatesShape($tmpdStates,'backward');
        }

        $dInputs = $this->differentiate($dOutputs);
        if(!$this->shapeInspection) {
            $tmpdStates = $dInputs;
            $tmpdInputs = array_shift($tmpdStates);
            if(count($tmpdStates)>0) {
                $this->assertStatesShape($tmpdStates,'backward');
            }
            $this->assertInputShape($tmpdInputs,'backward');
        }
        $this->collectGradients($grads,$oidsToCollect);

        return $dInputs;
    }

    protected function call(array $inputs,bool $training)
    {
        $K = $this->backend;
        $container = $this->container();
        $initialStates = $inputs;
        $inputs = array_shift($initialStates);
        $container->enableInitialStates=(count($initialStates)>0)?true:false;
        [$batches,$timesteps,$feature]=$inputs->shape();
        if(count($initialStates)==0 && $this->stateful) {
            $initialStates = $this->initialStates; // the statefull variable is not in container
        }
        if(count($initialStates)==0){
            foreach($this->statesShapes as $shape){
                $initialStates[] = $K->zeros(array_merge([$batches],$shape));
            }
        } else {
            $states = [];
            foreach($initialStates as $i => $s) {
                if($s===null) {
                    $shape = $this->statesShapes[$i];
                    $states[] = $K->zeros(array_merge([$batches],$shape));
                } else {
                    $states[] = $s;
                }
            }
            $initialStates = $states;
            unset($states);
        }
        
        $outputs = null;
        if($this->returnSequences){
            $outputs = $K->zeros([$batches,$timesteps,$this->units]);
        }
        [$outputs,$states,$calcStates] = $K->rnn(
            [$this->cell,'forward'],
            $inputs,
            $initialStates,
            $training,
            $outputs,
            $this->goBackwards
        );
        $container->calcStates = $calcStates;
        $container->origInputsShape = $inputs->shape();
        if($this->stateful) {
            $this->initialStates = $states; // the statefull variable is not in container
        }
        if($this->returnState){
            return array_merge([$outputs],$states);
        } else {
            return [$outputs];
        }
    }

    protected function differentiate(array $dOutputs)
    {
        $K = $this->backend;
        $container = $this->container();
        $dNextStates = $dOutputs;
        $dOutputs = array_shift($dNextStates);

        $dInputs=$K->zeros($container->origInputsShape);
        if(count($dNextStates)==0){
            $batches = $dOutputs->shape()[0];
            foreach($this->statesShapes as $shape){
                $dNextStates[] = $K->zeros(array_merge([$batches],$shape),$dOutputs->dtype());
            }
        }

        $grads = $this->cell->getGrads();
        foreach($grads as $grad){
            $K->clear($grad);
        }
        [$dInputs,$dPrevStates] = $K->rnnBackward(
            [$this->cell,'backward'],
            $dOutputs,
            $dNextStates,
            $container->calcStates,
            $dInputs,
            $this->goBackwards
        );
        $container->calcStates = null;
        if($container->enableInitialStates) {
            return array_merge([$dInputs], $dPrevStates);
        } else {
            return [$dInputs];
        }
    }

    /**
    *  @param Variable  $inputs
    *  @param bool      $training
    *  @param array<Variable> $initialStates
    *  @param array     $options
    *  @return array<Variable>
    *       outputs
    */
    public function __invoke($inputs, $training, array $initialStates=null, array $options=null)
    {
        $outputs = null;
        if($this->outputShape==null) {
            $inputShape = null;
            $creator = $inputs->creator();
            if($creator) {
                $inputShape = [$inputs];
            }
            $outputs = $this->build($inputShape);
        }
        if($inputs instanceof Undetermined) {
            if($outputs===null) {
                throw new InvalidArgumentException('Undetermined is found in second calling.');
            }
            if(is_array($outputs)) {
                $states = $outputs;
                $outputs = array_shift($states);
                return [$outputs,$states];
            } else {
                return $outputs;
            }
        }

        $inputsVariables = [$inputs];
        if($initialStates!==null) {
            $rawStatus = array_map(function($stat){return $stat->value();},$initialStates);
            $inputsVariables = array_merge($inputsVariables,$initialStates);
        } else {
            $rawStatus = null;
        }
        $session = $this->preGradientProcessOnSession($inputsVariables,['training'=>$training]);
        $inputs = $inputs->value();
        if(!is_bool($training)) {
            $training = $training->value();
        }
        $session->begin();
        try {
            $outputs = $this->forward($inputs,$training,$rawStatus,$options);
        } catch(Throwable $e) {
            $session->end();
            throw $e;
        }
        $session->end();

        if(is_array($outputs)) {
            [$o, $outputs] = $outputs;
            array_unshift($outputs, $o);
        } else {
            $outputs = [$outputs];
        }
        $outputsVariables = $this->postGradientProcessOnSession(
            $this->backend, $session, $inputsVariables, $outputs);
        
        if(count($outputsVariables)>1) {
            $outputs = array_shift($outputsVariables);
            return [$outputs,$outputsVariables];
        } else {
            return $outputsVariables[0];
        }
    }

    /**
     * Call from SessionFunc in compiled graph
     */
    public function _rawCall(array $inputs,array $options)
    {
        $training = $options['training'] ?? false;
        $results = $this->call($inputs,$training);
        return $results;
    }

    public function __clone()
    {
        if(isset($this->cell)) {
            $this->cell = clone $this->cell;
        }
    }
}
