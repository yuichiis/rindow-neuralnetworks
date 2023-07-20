<?php
namespace Rindow\NeuralNetworks\Loss;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\Variable;
use Rindow\NeuralNetworks\Gradient\Core\GradientTape;
use InvalidArgumentException;
use DomainException;
use ArrayAccess;

abstract class AbstractCrossEntropy extends AbstractLoss implements Loss//,Activation
{
}
