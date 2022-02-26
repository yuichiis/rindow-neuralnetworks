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
    protected $value;
    protected $grad;
    protected $creator;
    protected $generation=0;
    protected $name;

    public function __construct(object $backend, $value, $options=null)
    {
        extract($this->extractArgs([
            'name'=>null,
            'dtype'=>null,
            'reference'=>null,
        ],$options));
        $this->backend = $backend;
        if($value instanceof VariableInterface) {
            $value = $value->value();
        }
        if($value instanceof NDArray) {
            if($reference) {
                $this->value = $value;
            } else {
                $this->value = $backend->copy($value);
            }
        } elseif(is_bool($value)) {
            $this->value = $value;
        } elseif($value===null) {
            $this->value = null;
        } elseif(is_array($value)||is_numeric($value)) {
            $this->value = $backend->array($value);
        } else {
            throw InvalidArgumentException('Invalid vaule type:'.gettype($value));
        }
        $this->name = $name;
    }

    public function value()
    {
        return $this->value;
    }

    public function assign($value) : void
    {
        $backend = $this->backend;
        if($value instanceof VariableInterface) {
            $value = $value->value();
        }
        if($value instanceof NDArray) {
            $backend->copy($value,$this->value);
        } elseif(is_bool($value)) {
            if(!is_bool($this->value)) {
                throw InvalidArgumentException('vaule types must be equal:'.gettype($value));
            }
            $this->value = $value;
        } elseif(is_array($value)||is_numeric($value)) {
            $backend->copy($backend->array($value),$this->value);
        } else {
            throw InvalidArgumentException('Invalid vaule type:'.gettype($value));
        }
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
        if($this->value===null) {
            throw new LogicException('Variable has no value');
        }
        return $this->value->dtype();
    }

    public function shape()
    {
        if($this->value===null) {
            throw new LogicException('Variable has no value');
        }
        return $this->value->shape();
    }

    public function ndim()
    {
        if($this->value===null) {
            throw new LogicException('Variable has no value');
        }
        return $this->value->ndim();
    }

    public function size()
    {
        if($this->value===null) {
            throw new LogicException('Variable has no value');
        }
        return $this->value->size();
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
