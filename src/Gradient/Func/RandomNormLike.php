<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;
use Rindow\NeuralNetworks\Gradient\Core\NullValue;

class RandomNormLike extends AbstractFunction
{
    protected float $mean;
    protected float $scale;
    protected ?int $seed;

    public function __construct(
        object $backend,
        ?float $mean=null,
        ?float $scale=null,
        ?int $seed=null,
        ?string $name=null,
    )
    {
        $mean ??= 0.0;
        $scale ??= 1.0;
        parent::__construct($backend,name:$name);
        $this->mean = $mean;
        $this->scale = $scale;
        $this->seed = $seed;
    }

    protected function call(array $inputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $shape = $inputs[0]->shape();
        $dtype = $inputs[0]->dtype();
        $outputs = $K->randomNormalVariables($shape,$this->mean,$this->scale,$dtype,$this->seed);
        $container->shape = $shape;
        $container->dtype = $dtype;
        $this->unbackpropagatables = [true];
        return [$outputs];
    }

    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $dInputs = [new NullValue()];
        return $dInputs;
    }
}
