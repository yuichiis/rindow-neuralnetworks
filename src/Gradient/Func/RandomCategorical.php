<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;
use Rindow\NeuralNetworks\Gradient\Core\NullValue;

class RandomCategorical extends AbstractFunction
{
    protected ?int $numSamples;
    protected bool $softmax;
    protected int $dtype;
    protected ?int $seed;

    public function __construct(
        object $backend,
        ?int $numSamples=null,
        ?bool $softmax=null,
        ?int $dtype=null,
        ?int $seed=null,
        ?string $name=null,
    )
    {
        $softmax ??= true;
        $dtype ??= NDArray::int32;
        parent::__construct($backend,name:$name);
        $this->numSamples = $numSamples;
        $this->softmax = $softmax;
        $this->dtype = $dtype;
        $this->seed = $seed;
    }

    protected function call(array $inputs) : array
    {
        $K = $this->backend;
        $container = $this->container();

        $logits = $inputs[0];
        $batchShape = $logits->shape();
        if(count($batchShape)==0) {
            throw new Exception("More than one dimension is required.");
        }
        $singleLogits = (count($batchShape)==1);
        $numActions = array_pop($batchShape);
        $batchDims = count($batchShape);
        $batchSize = array_product($batchShape);
        $logits = $logits->reshape([$batchSize,$numActions]);
        if($this->softmax) {
            $probs = $K->softmax($logits);
        } else {
            $probs = $K->exp($logits);
        }
        if($this->numSamples!==null && $singleLogits) {
            $probs = $probs->reshape([$probs->size()]);
        }
        $outputs = $K->randomCategorical(
            $probs,
            numSamples:$this->numSamples,
            dtype:$this->dtype,
            seed:$this->seed
        );
        if($this->numSamples===null) {
            $outputs = $outputs->reshape($batchShape);
        }
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
