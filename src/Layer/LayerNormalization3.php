<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use LogicException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class LayerNormalization extends AbstractNormalization
{
    public function __construct(
        object $backend,
        int $axis=null,
        float $epsilon=null,
        bool $center=null,
        bool $scale=null,
        string|callable $beta_initializer=null,
        string|callable $gamma_initializer=null,
        string $name=null,
    )
    {
        parent::__construct(
            $backend,
            $axis,
            $epsilon,
            $center,
            $scale,
            $beta_initializer,
            $gamma_initializer,
        );
        // defaults
        $name = $name ?? null;

        $this->initName($name,'layernormalization');
        $this->allocateWeights(2);
    }

    protected function buildNoTrainingMode(array $kernelShape) : void
    {
    }

    public function reverseSyncWeightVariables() : void
    {
        $this->beta = $this->weights[0]->value();
        $this->gamma = $this->weights[1]->value();
    }

    public function getConfig() : array
    {
        return [
            'options' => [
                'axis'=>$this->axis,
                'epsilon'=>$this->epsilon,
                'beta_initializer'=>$this->betaInitializerName,
                'gamma_initializer'=>$this->gammaInitializerName,
            ]
        ];
    }

    public function getParams() : array
    {
        return [$this->beta,$this->gamma];
    }

    public function getGrads() : array
    {
        return [$this->dBeta,$this->dGamma];
    }

    protected function call(NDArray $inputs, bool $training=null) : NDArray
    {
        echo "============= call ============================\n";
        $K = $this->backend;
        $mo = $this->backend->localMatrixOperator();
        //if($training===null) {
        //    throw new InvalidArgumentException("training option must be true or false.");
        //}
        $container = $this->container();
        // (batch,heads...,feature) => (batch*heads,feature)
        $inputs = $this->transformShape($inputs);

        // normalization
        // xn = (x - mean(x)) / sqrt(mean( (x - mean(x))**2 ) + eps)
        //

        // mean = mean(x)
        // center = x - mean(x)
        $mean = $K->mean($inputs,axis:-1);                          // (batch*heads)
        echo "mean=".$mo->shapeToString($mean->shape())."\n";
        $center_x = $K->sub($inputs, $mean, trans:true);            // (batch*heads,feature)
        echo "sub=".$mo->shapeToString($center_x->shape())."\n";

        // variance = mean(square(x - mean), axis=-1)
        $variance = $K->mean($K->square($center_x), axis:-1);       // (batch*heads)
        echo "variance=".$mo->shapeToString($variance->shape())."\n";

        // std = sqrt(variance+eps)
        // normalized_x = x-mean(x) / std
        $std = $K->sqrt($K->increment($variance, $this->epsilon));  // (batch*heads)
        $norm_x = $K->div($center_x, $std, trans:true);             // (batch*heads,feature)

        $container->xc = $center_x; // (batch*head,feature)
        $container->xn = $norm_x;   // (batch*head,feature)
        $container->std = $std;     // (batch*heads)
        $container->variance = $variance;     // (batch*heads)

        if($this->gamma) {
            $outputs = $K->mul($this->gamma, $norm_x);
        } else {
            $outputs = $norm_x;
        }
        if($this->beta) {
            $outputs = $K->add($outputs, $this->beta);
        }

        $outputs = $this->untransformShape($outputs);
        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        echo "============= differentiate ============================\n";
        $mo = $this->backend->localMatrixOperator();
        $K = $this->backend;
        $dOutputs = $this->transformShape($dOutputs);
        $container = $this->container();
        $center_x = $container->xc;         // (batch*head,feature)
        $norm_x = $container->xn;           // (batch*head,feature)
        $std = $container->std;             // (batch*heads)
        $variance = $container->variance;   // (batch*heads)

        $tmp = $dOutputs->shape();
        $feature_dim = array_pop($tmp);

        // d_scaled_x = dOutputs                    // (batch*head,feature)
        // d_norm_x = d_scaled_x * gamma            // (batch*head,feature)
        if($this->dBeta) {
            $dbeta = $K->sum($dOutputs,axis:0,output:$this->dBeta);
            //$K->copy($dbeta,$this->dBeta);
        }
        if($this->dGamma) {
            $dgamma = $K->sum($K->mul($norm_x, $dOutputs), axis:0, output:$this->dGamma);
            //$K->copy($dgamma,$this->dGamma);
            $d_norm_x = $K->mul($this->gamma, $dOutputs);    // (batch*head,feature)
        } else {
            $d_norm_x = $dOutputs;                           // (batch*head,feature)
        }
        if($std===null) {
            throw new LogicException('not initialized for training');
        }
        // d_std = -sum(d_norm_x*center_x / (std**2)) :: (std**2)=(variance+epsilon)
        $d_std = $K->scale(-1.0, $K->sum(                               // (batch*head)
            $K->div($K->mul($d_norm_x, $center_x), $K->increment($variance, $this->epsilon), trans:true),// (batch*head,feature)
            axis:-1
        ));
        // d_variance = (d_std/2) / std
        $d_variance = $K->div($K->scale(0.5, $d_std), $std);            // (batch*head)
        // d_center_x = d_normalized_x / std
        $d_center_x = $K->div($d_norm_x, $std, trans:true);             // (batch*head,feature)
        // d_x = d_center_x + 2*center_x*d_variance/feature_dim - sum(d_center_x)/feature_dim
        //$K->update_add($d_center_x,                                     // (batch*head,feature)
        //    $K->scale(2.0/$feature_dim, $K->mul($center_x, $d_variance, trans:true))
        //);
        //$dInputs = $K->sub(                                             // (batch*head,feature)
        //    $d_center_x,                                                // (batch*head,feature)
        //    $K->scale(1/$feature_dim,$K->sum($d_center_x, axis:-1)),    // (batch*head)
        //    trans:true,
        //);
        // d_x = d_center_x - 1/feature_dim*(-2*center_x*d_variance + sum(d_center_x))
        $dInputs = $K->sub(                                             // (batch*head,feature)
            $d_center_x,                                                // (batch*head,feature)
            $K->scale(                                                  // (batch*head,feature)
                1/$feature_dim,                                         //  scaler
                $K->add(                                                // (batch*head,feature)
                    $K->scale(                                          // (batch*head,feature)
                        -2,                                             //  scaler
                        $K->mul(                                        // (batch*head,feature)
                            $center_x,                                  // (batch*head,feature)
                            $d_variance,                                // (batch*head)
                            trans:true
                        )
                    ),
                    $K->sum($d_center_x,axis:-1),                       // (batch*head)
                    trans:true
                )
            )
        );
        // d_x = d_center_x + 2/feature_dim*center_x*d_variance - 1/feature_dim*sum(d_center_x)
        // d_x = d_center_x - 1/feature_dim*(-2*center_x*d_variance + sum(d_center_x))
        // d_x = (d_scaled_x*gamma)/sqrt(variance+eps)
        //       - 1/feature_dim*
        //          (-2*(x-sum(x)/feature_dim)
        //              *
        //              ((
        //                  -sum(
        //                      (d_scaled_x * gamma)*(x-sum(x)/feature_dim)
        //                          / 
        //                      (variance+eps)
        //                  )
        //                      /
        //                  2
        //              ) / sqrt(variance+eps))
        //              + 
        //              sum(d_scaled_x*gamma/sqrt(variance+eps))
        //          )

        $dInputs = $this->untransformShape($dInputs);
        return $dInputs;
    }

    public function __clone()
    {
        if(isset($this->gamma)) {
            $this->gamma = clone $this->gamma;
        }
        if(isset($this->beta)) {
            $this->beta = clone $this->beta;
        }
        if(isset($this->dGamma)) {
            $this->dGamma = clone $this->dGamma;
        }
        if(isset($this->dBeta)) {
            $this->dBeta = clone $this->dBeta;
        }

        $this->allocateWeights(2);
        if($this->assignedWeights) {
            $this->syncWeightVariables();
        }
    }
}
