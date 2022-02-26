<?php
namespace Rindow\NeuralNetworks\Gradient\Core;

use WeakReference;

class VariableReference
{
    //protected $oid;
    protected $ref;
    protected $shape;
    protected $dtype;

    public function __construct(Variable $variable)
    {
        //$this->oid = spl_object_id($variable);
        $this->ref = WeakReference::create($variable);
        $value = $variable->value();
        if($value!==null &&
           !($value instanceof UndeterminedNDArray &&
             $value->isNull())) {
            $this->shape = $value->shape();
            $this->dtype = $value->dtype();
        }
    }

    //public function oid()
    //{
    //    return $this->oid;
    //}

    public function ref()
    {
        return $this->ref;
    }
    public function get()
    {
        return $this->ref->get();
    }

    public function shape()
    {
        return $this->shape;
    }

    public function dtype()
    {
        return $this->dtype;
    }

    public function valueShape()
    {
        if($this->shape===null) {
            return null;
        }
        $shape = $this->shape;
        array_shift($shape);
        return $shape;
    }
}
