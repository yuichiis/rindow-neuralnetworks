<?php
declare(strict_types=1);
namespace Rindow\NeuralNetworks\Gradient\Core;

use InvalidArgumentException;
use LogicException;
use Throwable;
use Rindow\NeuralNetworks\Support\Control\Context;
use Rindow\NeuralNetworks\Layer\LayerBase;
use Rindow\NeuralNetworks\Gradient\Core\Undetermined;

class GradientTape implements Context
{
    use GraphUtils;

    static public $autoBackProp = null;
    static public $debugBackward = null;
    static public $debug = false;

    protected $backend;
    protected $persistent;
    protected $backup;
    protected $persistentGrads = [];
    protected $lockingObjects = [];

    public function __construct(object $backend,bool $persistent=null)
    {
        $this->backend = $backend;
        $this->persistent = $persistent;
    }

    public function enter() : void
    {
        echo "GradientTape enter\n";
        $this->backup = self::$autoBackProp;
        self::$autoBackProp = $this;
    }

    public function exit(Throwable $e=null) : bool
    {
        self::$autoBackProp = $this->backup;
        if(self::$autoBackProp) {
            self::$autoBackProp->lockObjects($this->lockingObjects);
        }
        echo "GradientTape exit\n";
        return false;
    }

    public function lockObjects(array $variables) : void
    {
        echo "lock ".implode(',',array_map('spl_object_id',$variables))."\n";
        $this->lockingObjects = array_merge($this->lockingObjects,$variables);
    }

    public function gradient($target,$sources)
    {
        if(self::$autoBackProp) {
            throw new LogicException("The gradient function is not supported for use within the automatic differentiation context.");
        }
        $K = $this->backend;
        $singleValue = false;
        if($target->creator()==null)
            return null;
        if(!is_array($sources)) {
            $singleValue = true;
            $sources = [$sources];
        }
        $gradients = [];

        $targetId = spl_object_id($target);
        if($this->persistent && array_key_exists($targetId,$this->persistentGrads)) {
            $grads = $this->persistentGrads[$targetId];
        } else {
            $grads = [];
            foreach($target->creator()->outputs() as $o) {
                $grads[$o->oid()] = $K->ones($o->shape(),$o->dtype());
            }
            //$grads[$targetId] = $K->onesLike($target->value());
        }

        $sourceIds = $this->getObjectIds($sources);
        if(!$this->persistent || !array_key_exists($targetId,$this->persistentGrads)) {
            $this->calcGradient($grads,$target,$sourceIds);
        }
        echo "select grads:";
        foreach ($sourceIds as $sourceId) {
            if(!array_key_exists($sourceId,$grads)) {
                throw new InvalidArgumentException("No applicable gradient found for source");
            }
            $gradients[] = $grads[$sourceId];
            echo $sourceId.":".$this->backend->toString($grads[$sourceId]).",";
        }
        echo "\n";
        if($this->persistent) {
            $this->persistentGrads[$targetId] = $grads;
        }

        if($singleValue) {
            return $gradients[0];
        }
        return $gradients;
    }

    protected function calcGradient(&$grads,$target,$sourceIds) : void
    {
        echo "locked objects: ".implode(',',array_map('spl_object_id',$this->lockingObjects))."\n";
        echo "====== start calcGradient =======\n";
        $graphOutputs = [$target];
        [$pipeline,$constants] = $this->buildPipeline($graphOutputs);
        $this->backwardPipeline($this->backend,$pipeline,$grads,$sourceIds);
        echo "result grads: ";
        foreach($grads as $oid => $g) {
            echo $oid.":".$this->backend->toString($g).",";
        }
        echo "\n";
        echo "====== end calcGradient =======\n";
    }
}
