<?php
declare(strict_types=1);
namespace Rindow\NeuralNetworks\Gradient\Core;

use InvalidArgumentException;
use Rindow\NeuralNetworks\Gradient\Module;

trait GraphUtils
{
    protected function buildPipeline(array $graphOutputs) : array
    {
        $funcs = array_map(function($o){return $o->creator();},$graphOutputs);
        usort($funcs,function($a,$b){return $a->generation()-$b->generation();});
        $pipeline = [];
        $constants = [];
        $used = [];
        foreach($funcs as $func) {
            $used[spl_object_id($func)] = true;
        }
        while(count($funcs)>0) {
            $func = array_pop($funcs);
            $pipeline[] = $func;
            foreach($func->inputs() as $input) {
                $creator = $input->creator();
                if($creator!=null) {
                    $oid = spl_object_id($creator);
                    if(!array_key_exists($oid,$used)) {
                        $used[$oid] = true;
                        $funcs[] = $creator;
                        usort($funcs,function($a,$b){return $a->generation()-$b->generation();});
                    }
                } else {
                    $constants[spl_object_id($input)] = $input->value(); 
                }
            }
        }
        //echo "=====built pipeline=====\n";
        //foreach (array_reverse($pipeline) as $func) {
        //    echo "  ".basename($func->className())."(";
        //    foreach ($func->inputs() as $value) {
        //        echo spl_object_id($value).",";
        //    }
        //    echo ")=";
        //    foreach ($func->outputs() as $value) {
        //        echo $value->oid().",";
        //    }
        //    echo "\n";
        //}
        //echo "====================\n";
        return [$pipeline,$constants];
    }

    public function backwardPipeline(
        object $backend,
        array $pipeline, array &$grads=null, array $oidsToCollect=null) : void
    {
        echo "==start backward pipeline==\n";
        //echo "initial grads:\n";
        //foreach($grads as $oid => $g) {
        //    echo "  oid:".$oid." [".implode(',',$g->shape())."]\n";
        //}

        $K = $backend;
        foreach($pipeline as $func) {
            echo "Start func:".basename($func->className())."(".basename(get_class($func)).")\n";
            $dOutputs = [];
            foreach($func->outputs() as $o) {
                $oid = $o->oid();
                if(isset($grads[$oid])) {
                    $dOutputs[] = $grads[$oid];
                    //echo "fetch oid ".$oid." :".$K->toString($grads[$oid])."\n";
                    // *** CAUTION ***
                    // Outputs are released as soon as the func object is
                    // released after being used in backwards.
                    // Index inconsistencies in grads occur because OIDs
                    // can be reused. Grads must be released to prevent
                    // this problem.
                    if(!is_array($oidsToCollect)) {
                        unset($grads[$oid]);
                        echo "unset oid".$oid."\n";
                    }
                } else {
                    //$shape = $o->valueShape();
                    //$dtype = $o->dtype();
                    //array_unshift($shape,$batchSize);
                    //$dOutputs[] = $K->zeros($shape(),$dtype());
                    $dOutputs[] = $K->zeros($o->shape(),$o->dtype());
                    echo "allocate oid ".$oid." zero\n";
                }
            }
            //echo "backward pipeline dOutputs:";
            //foreach ($dOutputs as $value) {
            //    echo "[".implode(',',$value->shape())."],";
            //}
            //echo "\n";
    
            echo "call ".basename($func->className())."::backward()\n";
            $tmpdInputs = $func->backward($dOutputs,$grads,$oidsToCollect);
            echo "return ".basename($func->className())."::backward()\n";
            echo "watch grads: ";
            foreach($grads as $oid => $g) {
                echo $oid.":".$K->toString($g).",";
            }
            echo "\n";
    
            unset($dOutputs);
            //echo "backward pipeline dInputs:";
            //foreach ($tmpdInputs as $value) {
            //    if($value===null) {
            //        echo "NULL,";
            //    } else {
            //        echo "[".implode(',',$value->shape())."],";
            //    }
            //}
            //echo "\n";

            $dDatas = array_map(null,$func->inputs(),$tmpdInputs);
            unset($tmpdInputs);

            foreach ($dDatas as [$input,$dx]) {
                $oid = spl_object_id($input);
                if(isset($grads[$oid])) {
                    // **** CAUTION ****
                    // Must create new Instance of NDArray
                    // Don't use "update_add"!
                    // Because sometime grad and dx are same instace.
                    // Using update_add causes problems when branching function output more than once.
                    $grads[$oid] = $K->add($grads[$oid],$dx);
                    echo "update oid ".$oid." :".$K->toString($grads[$oid])."\n";
                } else {
                    $grads[$oid] = $dx;
                    echo "attach oid ".$oid." :";
                    if($dx===null) {
                        echo "NULL";
                    } else {
                        echo $K->toString($dx);
                    }
                    echo "\n";
                }
            }
            echo "End func:".basename($func->className())."(".basename(get_class($func)).")\n";
        }
        echo "==end backward pipeline==\n";
    }

    protected function packVariables(object $backend,array $values) : array
    {
        return array_map(function($value) use ($backend) {
            return ($value!==null)?new Variable($backend,$value):null;
        },$values);
    }

    protected function unpackVariables(array $variables) : array
    {
        return array_map(function($variable){
            return ($variable instanceof Variable)?$variable->value():null;
        },$variables);
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

    protected function getObjectIds(array $variables)
    {
        return array_map(function($variable) {
            return ($variable!==null)?spl_object_id($variable):null;
        },$variables);
    }

    protected function repackVariables(object $backend,array $variables) : array
    {
        return array_map(function($variable) use ($backend) {
            return new Variable($backend,$variable);
        },$variables);
    }

    protected function setCreatorToVariables(object $creator,array $variables) : void
    {
        foreach($variables as $variable) {
            if($variable!==null) {
                $variable->setCreator($creator);
            }
        }
    }

}