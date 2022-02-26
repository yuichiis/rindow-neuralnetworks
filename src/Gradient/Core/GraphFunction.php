<?php
declare(strict_types=1);
namespace Rindow\NeuralNetworks\Gradient\Core;

use InvalidArgumentException;
use RuntimeException;
use Throwable;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\Control\Execute;
use Rindow\NeuralNetworks\Gradient\Module;

class GraphFunction
{
    use GraphUtils;

    const EAGER_EXECUTION = 0;
    const UNDER_CONSTRUCTION = 1;
    const EXECUTING = 2;

    static public $mode = self::EAGER_EXECUTION;

    protected $backupMode = [];

    /**
    *  @var object   backend
    */
    protected $backend;

    /**
    *  @var callable func
    */
    protected $func;

    /**
    *  @var bool built
    */
    protected $built = false;

    /**
    *  @var int   numOfInputs
    */
    protected $numOfInputs;

    /**
    *  @var int   numOfOutputs
    */
    protected $numOfOutputs;

    protected $startInputOids;

    protected $endOutputOids;

    /**
    *  @var array<AbstractFunction>   graph pipeline
    */
    protected $pipeline;

    /**
    *  @var Dict<NDArray>    constants for input oid in the graph
    */
    protected $constants;

    protected $alternateCreator;

    public function __construct(object $backend, callable $func, array $options=null)
    {
        $this->backend = $backend;
        $this->func = $func;
        $this->alternateCreator = $options['alternateCreator'] ?? null;
    }

    protected function executeOnMode($sessionFunc,int $mode,callable $func)
    {
        array_push($this->backupMode,self::$mode);
        self::$mode = $mode;
        try {
            $sessionFunc->begin();
            try {
                $outputs = $func();
            } catch(Throwable $e) {
                $sessionFunc->end();
                throw $e;
            }
            $sessionFunc->end();
        } catch(Throwable $e) {
            self::$mode = array_pop($this->backupMode);
            throw $e;
        }
        self::$mode = array_pop($this->backupMode);
        return $outputs;
    }

    /**
    *  @param array<Variable>  $inputs
    *       inputs
    *  @return array<Variable>
    *       outputs
    */
    public function __invoke(...$args)
    {
        if(!$this->built) {
            return $this->build($args);
        }
        $creator = $this->alternateCreator ?? $this;
        $sessionFunc = new GraphSession($creator,$args);
        if(count($args)!=$this->numOfInputs) {
            throw new InvalidArgumentException($this->numOfInputs.' arguments are required.');
        }
        if(self::$mode!=self::EXECUTING) {
            if(GradientTape::$autoBackProp) {
                $sessionFunc->_setGeneration($this->maxGeneration($args));
            }
            $args = $this->unpackVariables($args);
        }

        // execute graph 
        $outValues = $this->executeOnMode($sessionFunc,self::EXECUTING,function() use ($args) {
            return $this->_rawCall($args,[]);
        });

        // finalize outputs
        if(self::$mode!=self::EXECUTING) {
            $outputs = $this->packVariables($this->backend,$outValues);
            if(GradientTape::$autoBackProp) {
                $this->setCreatorToVariables($sessionFunc,$outputs);
                $sessionFunc->_setOutputsVariables($this->referenceVariables($outputs));
            }
        }
        if(count($outputs)==1) {
            $outputs = $outputs[0];
        }
        return $outputs;
    }

    public function _rawCall(array $inputs,array $options) : array
    {
        $vars = $this->constants;
        foreach(array_map(null,$this->startInputOids,$inputs) as $d) {
            [$oid,$inp] = $d;
            $vars[$oid] = $inp;
        }
        $funcs = $this->pipeline;

        foreach($funcs as $func) {
            $oids = $this->getObjectIds($func->inputs());
            $inps = array_map(function($oid) use ($vars) {return $vars[$oid];}, $oids);
            $opts = [];
            foreach ($func->options() as $key => $variable) {
                $oid = spl_object_id($variable);
                $opts[$key] = $vars[$oid] ?? null;
            }
            $outs = $func->_rawCall($inps,$opts);
            foreach(array_map(null,$func->outputs(),$outs) as $d) {
                [$o,$out] = $d;
                $vars[$o->oid()] = $out;
            }
        }
        
        $outValues = [];
        foreach ($this->endOutputOids as $oid) {
            $outValues[] = $vars[$oid];
        }
        return $outValues;
    }

    protected function build(array $inputs)
    {
        $K = $this->backend;
        echo "==BUILD START==\n";
        $creator = $this->alternateCreator ?? $this;
        $sessionFunc = new GraphSession($creator,$inputs);
        $sessionFunc->_setGeneration($this->maxGeneration($inputs));
        echo "input oids=[".implode(',',array_map(function($x){return spl_object_id($x);},$inputs))."]\n";
        $inputs = $this->repackVariables($this->backend,$inputs);
        echo "repack input oids=[".implode(',',array_map(function($x){return spl_object_id($x);},$inputs))."]\n";
        $this->startInputOids = $this->getObjectIds($inputs);
        echo "input values=[".implode(',',array_map(function($x) use ($K) {return $K->toString($x->value());},$inputs))."]\n";

        // build graph
        $this->numOfInputs = count($inputs);
        if($inputs[0] instanceof Undetermined) {
            for($i=0;$i<$this->numOfOutputs;$i++) {
                $graphOutputs[] = new Undetermined();
            }
        } else {
            $func = $this->func;
            $graphOutputs = Execute::with(new GradientTape($this->backend),function() use ($sessionFunc,$func,$inputs) {
                return $this->executeOnMode($sessionFunc,self::UNDER_CONSTRUCTION,function() use ($func,$inputs) {
                    return $func(...$inputs);
                });
            });
            if(!is_array($graphOutputs)) {
                $graphOutputs = [$graphOutputs];
            }
        }

        echo "graphOutputs oids=[".implode(',',array_map(function($x){return spl_object_id($x);},$graphOutputs))."]\n";
        echo "output values=[".implode(',',array_map(function($x) use ($K) {return $K->toString($x->value());},$graphOutputs))."]\n";
        $this->endOutputOids = $this->getObjectIds($graphOutputs);

        [$pipeline,$constants] = $this->buildPipeline($graphOutputs);

        $this->constants = $constants; // NDArray
        $this->pipeline = array_reverse($pipeline); // Func
        $this->built = true;
        $outputs = $this->repackVariables($this->backend,$graphOutputs);
        foreach($pipeline as $func) {
            foreach($func->inputs() as $o) {
                if(!($o->creator()==null && !in_array(spl_object_id($o),$this->startInputOids))) {
                    // Clearing variables without constants and weights.
                    // Because It wants to save the math buffer.
                    $o->_clearValue();
                }
            }
        }
        $this->setCreatorToVariables($sessionFunc,$outputs);
        $sessionFunc->_setOutputsVariables($this->referenceVariables($outputs));
        echo "==BUILD END==\n";
        if(count($outputs)==1) {
            return $outputs[0];
        }
        return $outputs;
    }

    /**
    *  @param array<NDArray>  $inputs
    *       inputs
    *  @return array<NDArray>
    *       outputs
    */
    public function backward(array $dOutputs, array &$grads=null, array $oidsToCollect=null) : array
    {
        if(!$this->built) {
            throw new RuntimeException('Not yet built');
        }
        $K = $this->backend;
        $backupGradOids = array_keys($grads);
        foreach(array_map(null,$this->endOutputOids,$dOutputs) as $oset) {
            [$oid,$dOut] = $oset; 
            $grads[$oid] = $dOut;
        }
        unset($output);
        unset($dOut);
        unset($oset);
        unset($dOutputs);

        $pipeline = array_reverse($this->pipeline);
        $this->backwardPipeline($this->backend, $pipeline, $grads, $oidsToCollect);

        foreach($this->startInputOids as $oid) {
            if(!isset($grads[$oid])) {
                //throw new InvalidArgumentException("Invalid input variables");
                $dInputs[] = null; // maybe, it is skiped argument in internal.
                continue;
            }
            $dInputs[] = $grads[$oid];
        }
        $unsets = [];
        foreach ($grads as $oid => $value) {
            if(!in_array($oid,$oidsToCollect) && !in_array($oid,$backupGradOids)) {
                $unsets[] = $oid;
            }
        }
        foreach ($unsets as $oid) {
            unset($grads[$oid]);
        }
        return $dInputs;
    }

}
