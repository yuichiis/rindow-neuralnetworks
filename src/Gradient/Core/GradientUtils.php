<?php
declare(strict_types=1);
namespace Rindow\NeuralNetworks\Gradient\Core;

use InvalidArgumentException;
use stdClass;

trait GradientUtils
{
    protected $container;

    protected function postGradientProcess(
        $backend, array $inputsVariables, array $outputs) : array
    {
        $outputsVariables = [];
        foreach ($outputs as $v) {
            if($v === null) {
                $outputsVariables[] = new Undetermined();
            } else {
                $outputsVariables[] = new Variable($backend,$v);
            }
        }
        if(GradientTape::$autoBackProp) {
            $this->inputsVariables = $inputsVariables;
            $this->generation = $this->maxGeneration($inputsVariables);
            foreach ($outputsVariables as $o) {
                $o->setCreator($this);
            }
            $this->outputsVariables = $this->referenceVariables($outputsVariables);
            $this->lockVariableObjects($outputsVariables);
        }
        return $outputsVariables;
    }

    protected function preGradientProcessOnSession($inputsVariables,$optionsVariables=null) : object
    {
        $session = new GraphSession($this,$inputsVariables,$optionsVariables=null);
        $session->_setGeneration($this->maxGeneration($inputsVariables));
        return $session;
    }

    protected function postGradientProcessOnSession(
        object $backend, object $session, array $inputsVariables, array $outputs) : array
    {
        $outputsVariables = [];
        foreach ($outputs as $v) {
            if($v === null) {
                $outputsVariables[] = new Undetermined();
            } else {
                $outputsVariables[] = new Variable($backend,$v);
            }
        }
        if(GradientTape::$autoBackProp) {
            $this->setCreatorToVariables($session,$outputsVariables);
            $session->_setOutputsVariables($this->referenceVariables($outputsVariables));
            $this->lockVariableObjects($outputsVariables);
        }
        return $outputsVariables;
    }

    protected function container() : object
    {
        $session = GraphSession::$session;
        if($session==null) {
            if($this->container===null) {
                $this->container = new stdClass();
            }
            return $this->container;
        }
        return $session->localContainer($this);
    }

    protected function packVariable(object $backend, $value)
    {
        if($value instanceof Variable) {
            return $value;
        } 
        return new Variable($backend,$value);
    }

    protected function packVariables(object $backend,array $values) : array
    {
        return array_map(function($value) use ($backend) {
            return ($value!==null)?new Variable($backend,$value):null;
        },$values);
    }

    protected function referenceVariables(array $variables) : array
    {
        return array_map(function($variable) {
            return ($variable!==null)?$variable->reference():null;
        },$variables);
    }

    protected function maxGeneration(array $variables)
    {
        return array_reduce($variables,function($max,$variable) {
            return ($variable!==null)?max($max,$variable->generation()):$max;},-1);
    }

    protected function setCreatorToVariables(object $creator,array $variables) : void
    {
        foreach($variables as $variable) {
            if($variable!==null) {
                $variable->setCreator($creator);
            }
        }
    }

    protected function lockVariableObjects(array $variables) : void
    {
        if(GradientTape::$autoBackProp) {
            GradientTape::$autoBackProp->lockObjects($variables);
        }
    }
}
