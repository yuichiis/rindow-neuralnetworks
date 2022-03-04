<?php
declare(strict_types=1);
namespace Rindow\NeuralNetworks\Gradient\Core;

use InvalidArgumentException;
use LogicException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;
use Rindow\NeuralNetworks\Gradient\Variable as VariableInterface;

class Variable implements VariableInterface
{
    use GenericUtils;
    protected $backend;
    protected $undetermined;
    protected $reference;
    protected $name;
    protected $value;
    protected $creator;
    protected $generation=0;

    public function __construct(object $backend, $value, $options=null)
    {
        extract($this->extractArgs([
            'name'=>null,
            'reference'=>null,
            'undetermined'=>null,
        ],$options));
        $this->backend = $backend;
        $this->undetermined = $undetermined;
        $this->name = $name;
        $this->reference = $reference;
        if(!$undetermined) {
            $this->assign($value);
        }
    }

    public function assign($value) : void
    {
        if($value instanceof VariableInterface) {
            $value = $value->value();
        }
        if($value instanceof NDArray) {
            if($this->reference) {
                $this->value = $value;
            } else {
                $this->value = $this->backend->copy($value);
            }
        } elseif(is_bool($value)) {
            $this->value = $value;
        } elseif(is_array($value)||is_numeric($value)) {
            $this->value = $this->backend->array($value);
        } else {
            throw InvalidArgumentException('Invalid vaule type:'.gettype($value));
        }
        $this->undetermined = false;
    }

    public function isUndetermined() : bool
    {
        return $this->undetermined;
    }

    public function value()
    {
        if($this->undetermined) {
            throw new LogicException("Undetermined variable");
        }
        return $this->value;
    }

    public function name()
    {
        return $this->name;
    }

    public function setName($name)
    {
        return $this->name = $name;
    }

    public function dtype()
    {
        $value = $this->value;
        if($value===null) {
            throw new LogicException('Variable has no value');
        }
        if($value instanceof NDArray) {
            return $value->dtype();
        }
        if(is_bool($value)) {
            return NDArray::bool;
        }
        throw new RuntimeException('invalid type:'.(is_object($value)?get_class($value):gettype($value)));
    }

    public function shape()
    {
        $value = $this->value;
        if($value===null) {
            throw new LogicException('Variable has no value');
        }
        if(is_bool($value)) {
            return [];
        }
        return $value->shape();
    }

    public function ndim()
    {
        $value = $this->value;
        if($value===null) {
            throw new LogicException('Variable has no value');
        }
        if(is_bool($value)) {
            return 0;
        }
        return $value->ndim();
    }

    public function size()
    {
        $value = $this->value;
        if($value===null) {
            throw new LogicException('Variable has no value');
        }
        if(is_bool($value)) {
            return 0;
        }
        return $value->size();
    }

    /**
    * @return Function $creator
    *   creater function
    */
    public function creator()
    {
        return $this->creator;
    }

    /**
    * @param Function $creator
    *   creater function
    * @return void
    */
    public function setCreator($creator) : void
    {
        $this->creator = $creator;
        $this->generation = $creator->generation() + 1;
    }

    public function generation() : int
    {
        return $this->generation;
    }

    public function valueShape()
    {
        if($this->value===null) {
            return null;
        }
        $shape = $this->shape();
        array_shift($shape);
        return $shape;
    }

    public function reference()
    {
        return new VariableReference($this);
    }

    public function _clearValue()
    {
        $this->value = null;
    }
}
