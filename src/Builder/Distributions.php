<?php
namespace Rindow\NeuralNetworks\Builder;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Distribution\Distribution;
use Rindow\NeuralNetworks\Distribution\Categorical;
use Rindow\NeuralNetworks\Distribution\Normal;
use Rindow\NeuralNetworks\Gradient\Variable;

class Distributions
{
    protected object $builder;

    public function __construct(object $builder)
    {
        $this->builder = $builder;
    }

    public function Categorical(
        ?Variable $logits=null,
        ?Variable $probs=null,
        ) : Distribution
    {
        return new Categorical($this->builder, logits:$logits,probs:$probs);
    }

    public function Normal(
        ?Variable $loc=null,
        ?Variable $scale=null,
        ) : Distribution
    {
        return new Normal($this->builder, loc:$loc,scale:$scale);
    }
}
